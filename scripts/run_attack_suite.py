#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/run_attack_suite.py
===========================

E3-DoH Reproducibility — attack-suite driver.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This script runs the fifteen attack simulations of Chapter 5 — the
five Table 18 scenarios, the five Table 22 scenarios, and the five
Table 23 scenarios — in one invocation, collecting their outcomes
into a single report. It is the primary driver for the reviewer who
wants to exercise the whole attack surface in one command, rather
than running each of the five attack modules individually.

The five attack modules (dns_tunneling, timing_side_channel,
quic_congestion, cert_injection, padding_oracle) each expose a
simulator class whose `run()` method returns a report. This script
dispatches to the right simulator for each scenario, collects the
reports, and writes `outputs/attacks/suite_report.json`.

SAFETY GUARDS
-------------
The script does NOT itself touch the network. It invokes the
simulator's `run()` method, which enforces the four safety guards
from attack_suite_params.yaml §2 (isolation_required,
target_must_be_local, no_privileged_ports, no_external_network).
A guard violation raises `IsolationGuardViolation` or
`EgressDetectedError`, which the driver reports per-scenario.

The `--skip-guards` flag (which must be passed explicitly) tells the
driver to call `run(skip_guard_checks=True)`. This is the mode the
toy-mode reviewer uses; it is documented as "code-path exercise, no
network traffic generated". A reviewer who runs with `--skip-guards`
in full mode accepts that the simulation will construct packets but
will not actually send them (the driver's `_send_packets` is a no-op
when `skip_guard_checks=True`).

USAGE
-----
    # List the scenarios this script will run
    python scripts/run_attack_suite.py --list

    # Run the five Table 18 scenarios (the baseline detection ones)
    python scripts/run_attack_suite.py --group table_18

    # Run a single scenario
    python scripts/run_attack_suite.py --scenario A01

    # Run all fifteen scenarios
    python scripts/run_attack_suite.py --all

    # Toy-mode run (skips guards, does not send packets)
    python scripts/run_attack_suite.py --all --skip-guards

    # Continue past a scenario that fails (report it, run the rest)
    python scripts/run_attack_suite.py --all --skip-guards --continue-on-error

EXIT CODES
----------
    0  every requested scenario completed (with or without a fired trigger)
    1  at least one scenario raised (guard violation, config error, etc.)
    2  bad argument
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import (
    AttackMatrixError,
    ConfigError,
    load_attack_matrix,
    load_config,
    output_path,
)
from src.utils.logger import configure as configure_logger
from src.utils.logger import get_logger
from src.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# The scenario registry — which module handles which scenario.
# ---------------------------------------------------------------------------

# The five Table 18 scenarios.
_TABLE_18 = ("A01", "A02", "A03", "A04", "A05")

# The five Table 22 scenarios.
_TABLE_22 = ("A06", "A07", "A08", "A09", "A10")

# The five Table 23 scenarios.
_TABLE_23 = ("A11", "A12", "A13", "A14", "A15")

# The full registry: scenario id -> (module_name, class_name).
_REGISTRY: Dict[str, Tuple[str, str]] = {
    # A01, A06, A10, A11 → dns_tunneling
    "A01": ("src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
    "A06": ("src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
    "A10": ("src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
    "A11": ("src.attacks.dns_tunneling", "DNSTunnelingSimulator"),
    # A02 → timing_side_channel
    "A02": ("src.attacks.timing_side_channel", "TimingSideChannelSimulator"),
    # A03, A07, A13 → quic_congestion
    "A03": ("src.attacks.quic_congestion", "QUICCongestionSimulator"),
    "A07": ("src.attacks.quic_congestion", "QUICCongestionSimulator"),
    "A13": ("src.attacks.quic_congestion", "QUICCongestionSimulator"),
    # A04, A08, A09, A12, A14 → cert_injection
    "A04": ("src.attacks.cert_injection", "CertInjectionSimulator"),
    "A08": ("src.attacks.cert_injection", "CertInjectionSimulator"),
    "A09": ("src.attacks.cert_injection", "CertInjectionSimulator"),
    "A12": ("src.attacks.cert_injection", "CertInjectionSimulator"),
    "A14": ("src.attacks.cert_injection", "CertInjectionSimulator"),
    # A05, A15 → padding_oracle
    "A05": ("src.attacks.padding_oracle", "PaddingOracleSimulator"),
    "A15": ("src.attacks.padding_oracle", "PaddingOracleSimulator"),
}

_GROUPS = {
    "table_18": _TABLE_18,
    "table_22": _TABLE_22,
    "table_23": _TABLE_23,
}


# ---------------------------------------------------------------------------
# Simulator construction
# ---------------------------------------------------------------------------

def _build_simulator(
    scenario_id: str,
    config: Dict[str, Any],
    seed: int,
) -> Any:
    """
    Lazy-import the right module and construct the simulator class.

    Lazy import keeps the driver importable even when scapy is not
    installed (the user who runs --list does not need scapy).
    """
    if scenario_id not in _REGISTRY:
        raise ValueError(f"unknown scenario {scenario_id!r}")
    module_name, class_name = _REGISTRY[scenario_id]
    mod = __import__(module_name, fromlist=[class_name])
    cls = getattr(mod, class_name)
    return cls(config, scenario_id, seed=seed)


# ---------------------------------------------------------------------------
# Per-scenario execution
# ---------------------------------------------------------------------------

def _run_one(
    scenario_id: str,
    config: Dict[str, Any],
    *,
    skip_guards: bool,
    seed: int,
    log: Any,
) -> Dict[str, Any]:
    """
    Run one scenario and return a dict of its outcome. Never raises;
    on error, the returned dict has "error".
    """
    started = time.perf_counter()
    result: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "error": None,
    }
    try:
        sim = _build_simulator(scenario_id, config, seed)
        report = sim.run(skip_guard_checks=skip_guards)
        result["report"] = report.to_dict() if hasattr(report, "to_dict") else dict(report)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        log.warning(
            "scenario_failed",
            scenario=scenario_id, error=result["error"],
        )
    result["wall_time_s"] = round(time.perf_counter() - started, 3)
    return result


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_attack_suite.py",
        description="E3-DoH attack-suite driver.",
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true",
                   help="List the scenarios this script will run.")
    g.add_argument("--scenario", type=str, choices=sorted(_REGISTRY),
                   help="Run a single scenario.")
    g.add_argument("--group", type=str, choices=sorted(_GROUPS),
                   help="Run one of the thesis-table groups.")
    g.add_argument("--all", action="store_true",
                   help="Run all fifteen scenarios.")

    p.add_argument("--skip-guards", action="store_true",
                   help="Skip the safety guards (toy mode; no real traffic).")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going after a scenario raises.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a custom config file.")
    p.add_argument("--output", type=str, default=None,
                   help="Output path for the suite report. Default: "
                        "outputs/attacks/suite_report.json")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                   default="INFO")
    return p


def _print_list() -> None:
    print("Registered scenarios:")
    for group_name, members in _GROUPS.items():
        print(f"\n  {group_name}:")
        for sid in members:
            module, cls = _REGISTRY[sid]
            print(f"    {sid}  {module}.{cls}")
    print()


def _select_scenarios(args: argparse.Namespace) -> List[str]:
    if args.scenario:
        return [args.scenario]
    if args.group:
        return list(_GROUPS[args.group])
    if args.all:
        return sorted(_REGISTRY.keys())
    return []


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list:
        _print_list()
        return 0

    scenarios = _select_scenarios(args)
    if not scenarios:
        print("Error: no scenarios selected", file=sys.stderr)
        return 2

    # Load config.
    try:
        config = load_config(config_path=args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    config["runtime"]["mode"] = (
        "toy" if args.skip_guards else config["runtime"].get("mode", "toy")
    )

    seed = int(config.get("runtime", {}).get("seed", 42))
    set_global_seed(seed)

    log_cfg = config.get("logging", {}) or {}
    configure_logger(
        level=args.log_level,
        fmt=str(log_cfg.get("format", "json")),
        file_path=log_cfg.get("file", "outputs/logs/run_attack_suite.log"),
        console=True,
    )
    log = get_logger("e3doh.scripts.run_attack_suite")

    log.info(
        "suite_start",
        n_scenarios=len(scenarios),
        skip_guards=args.skip_guards,
        continue_on_error=args.continue_on_error,
    )

    # Load the attack matrix for cross-checking the trigger/action.
    try:
        matrix = load_attack_matrix()
    except AttackMatrixError as e:
        log.warning("attack_matrix_load_failed", error=str(e))
        matrix = None

    # Run each scenario.
    started = time.perf_counter()
    results: Dict[str, Dict[str, Any]] = {}
    for scenario_id in scenarios:
        log.info("scenario_start", scenario=scenario_id)
        result = _run_one(
            scenario_id, config,
            skip_guards=args.skip_guards, seed=seed, log=log,
        )

        # Cross-check the trigger/action against the attack matrix.
        if matrix is not None and result.get("report"):
            try:
                row = matrix[matrix["Attack_ID"] == scenario_id].iloc[0]
                rep = result["report"]
                result["matrix_consistent"] = (
                    rep.get("detection_trigger") == row["Detection_Trigger"]
                    and rep.get("mitigation_action") == row["Mitigation_Action"]
                )
            except (IndexError, KeyError) as e:
                result["matrix_consistent"] = None
                log.warning(
                    "matrix_cross_check_failed",
                    scenario=scenario_id, error=repr(e),
                )
        else:
            result["matrix_consistent"] = None

        results[scenario_id] = result

        if result.get("error"):
            log.warning(
                "scenario_completed_with_error",
                scenario=scenario_id, error=result["error"],
            )
            if not args.continue_on_error:
                break
        else:
            log.info(
                "scenario_completed",
                scenario=scenario_id,
                trigger_fires=result["report"].get("trigger_fires"),
                matrix_consistent=result.get("matrix_consistent"),
            )

    elapsed = round(time.perf_counter() - started, 3)

    # Summary.
    n_completed = sum(1 for r in results.values() if not r.get("error"))
    n_failed = sum(1 for r in results.values() if r.get("error"))
    n_matrix_consistent = sum(
        1 for r in results.values() if r.get("matrix_consistent") is True
    )
    n_trigger_fired = sum(
        1 for r in results.values()
        if r.get("report") and r["report"].get("trigger_fires")
    )

    # Write the report.
    output = {
        "suite": "run_attack_suite",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": config["runtime"]["mode"],
        "seed": seed,
        "run_id": config.get("_run_id"),
        "wall_time_s": elapsed,
        "skip_guards": args.skip_guards,
        "scenarios": results,
        "summary": {
            "n_requested": len(scenarios),
            "n_completed": n_completed,
            "n_failed": n_failed,
            "n_matrix_consistent": n_matrix_consistent,
            "n_trigger_fired": n_trigger_fired,
        },
    }

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = _REPO_ROOT / out_path
    else:
        out_path = output_path("attacks", "suite_report.json")

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
        log.info("suite_report_written", path=str(out_path))
    except OSError as e:
        log.error("suite_report_write_failed", path=str(out_path), error=str(e))
        return 1

    # Print summary.
    print()
    print(f"=== Attack suite complete "
          f"({config['runtime']['mode']} mode, "
          f"{'guards skipped' if args.skip_guards else 'guards enforced'}) ===")
    print(f"{'scenario':<10} {'status':<10} {'trigger':<10} {'matrix':<10}")
    for sid in scenarios:
        r = results.get(sid, {})
        if sid not in results:
            print(f"{sid:<10} {'not run':<10}")
            continue
        status = "ERROR" if r.get("error") else "OK"
        rep = r.get("report") or {}
        trigger = "fires" if rep.get("trigger_fires") else "no-fire"
        matrix = r.get("matrix_consistent")
        matrix_str = "match" if matrix else ("mismatch" if matrix is False else "n/a")
        print(f"{sid:<10} {status:<10} {trigger:<10} {matrix_str:<10}")
    print()
    print(f"Completed: {n_completed}/{len(scenarios)}")
    print(f"Failed:    {n_failed}")
    print(f"Triggers fired: {n_trigger_fired}/{n_completed}")
    print(f"Matrix consistent: {n_matrix_consistent}/{n_completed}")
    print(f"Wall time: {elapsed:.1f} s")
    print(f"Report: {out_path}")
    print()

    if n_failed > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
