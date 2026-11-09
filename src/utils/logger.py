#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/utils/logger.py
===================

E3-DoH Reproducibility — structured JSON logging.

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

This module centralises every log write in the artifact. It is the third
of the three utility modules (after math_utils.py and seed.py) and the
only module that both writes log records and *reads* them back for
verification. The verification script (`expected_outputs/verification_script.py`)
parses the run header to check that a run was made with the expected
config hashes; without this module, that check would have nothing to
read.

WHAT A LOG RECORD LOOKS LIKE
----------------------------
Every record is a single line of JSON with the following reserved keys
at the top level, plus a `fields` object for arbitrary caller-supplied
context:

    {
      "time": "2026-11-09T12:34:56.789Z",   # ISO 8601, UTC, millisecond precision
      "level": "INFO",                       # DEBUG | INFO | WARNING | ERROR
      "logger": "e3doh.train",               # logger name passed to get_logger()
      "event": "epoch_complete",             # the event name (first positional arg)
      "run_id": "a1b2c3d4e5f6a7b8",          # from config._run_id (optional)
      "pid": 12345,                          # process id
      "thread": "MainThread",                # thread name
      "fields": {                            # arbitrary caller-supplied context
        "epoch": 3,
        "loss": 0.042,
        "accuracy": 0.9987
      }
    }

The `fields` separation (rather than flattening everything to the top
level) is what makes `jq` queries robust: `.fields.epoch` will not
collide with a reserved key, and a caller who accidentally passes a
reserved key gets a clear error rather than a silent overwrite.

The `fields` object is optional (a record with no extra context still
has an empty `fields: {}`); the reserved keys are always present.

WHY JSON
--------
Three reasons:
  1. The verification script parses the log to check the run header.
     JSON parsing is `json.loads`; text parsing would require a regex
     and would be fragile.
  2. A reviewer who wants to extract all WARNING records from a run
     can do `jq 'select(.level == "WARNING")'` on the log, which is a
     one-liner. Text logs require `grep` plus manual parsing.
  3. The log is a reproducibility artifact; a reviewer who wants to
     reconstruct a run should be able to do so programmatically.

The one cost of JSON is that it is less pleasant to read on a terminal
than a traditional log line. For that reason the console handler
(when attached to a TTY) renders a text-friendly summary line while
the file always gets the full JSON record. This is the same split that
structlog and similar libraries use, and it lets a reviewer see what is
happening while still getting structured records in the file.

THE RUN HEADER
--------------
The first record a process writes to its log file is the *run header*,
a record with `event: "run_header"` that carries:

    - the full resolved config (with secrets redacted),
    - the SHA-256 of every config file that was loaded (from
      data_utils.load_config()["_hashes"]),
    - the git commit hash (if the checkout is a git working tree),
    - the Python version and the pinned-dependency versions,
    - the environment variables that affect reproducibility
      (PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG),
    - the seed state (from src.utils.seed.describe_seed_state, if
      seed.py is importable),
    - the run_id (from config._run_id).

The verification script reads the header to confirm that a fresh run
was made with the same config hashes as the committed expected
outputs. If the header is missing or the hashes differ, the run is not
comparable — that is a hard failure, not a warning.

The header is written once per process per run_id. If a process calls
get_logger() multiple times, the header is not re-written. If two
processes write to the same log file, each writes its own header;
`jq 'select(.event == "run_header")'` on the concatenated file returns
both.

REDACTION
---------
No secret should ever appear in a log record, but "should" is not the
same as "will". The logger applies a fixed redaction list to every
record: any key matching one of the patterns in REDACT_KEYS has its
value replaced with the literal string "[REDACTED]". This is a
belt-and-braces measure; the artifact is not supposed to log secrets
in the first place, and the redaction list is a last line of defense,
not a design principle.

The redaction is applied recursively to nested dicts and to dicts
inside lists. It does not redact string values (only values whose key
matches), because a caller who has a string field named `secret_note`
should get to keep it.

THREAD-SAFETY
-------------
The logger uses a per-instance threading.Lock around every file write.
On POSIX, a single f.write() of a line under PIPE_BUF (typically 4096
bytes) is atomic on a file opened in append mode, so the lock is
strictly speaking unnecessary for records under 4 KB. Records can
exceed 4 KB (a run header with 20 config hashes and a full config dump
easily exceeds it), and the lock makes concurrent writes safe for
those records too.

The lock is per-instance, not global. Two logger instances writing to
the same file (e.g. from two processes) are serialised by the OS's
O_APPEND semantics at the file-descriptor level, but interleaving of
*records* is still possible if a record exceeds PIPE_BUF. The artifact
does not use concurrent processes writing to the same log file; the
attack suite's parallel mode writes to separate files, and the run
header records the parallel mode setting so a reviewer knows.

FAILURE MODES
-------------
If the log file cannot be opened (permissions, missing parent
directory, disk full), the logger falls back to writing JSON to stderr
with a warning. It does NOT raise, because a logger that crashes
because its own output file is unwritable would obscure the actual
failure the caller is trying to log.

If the console handler fails (e.g. no TTY, no rich installed), the
logger silently falls back to plain print(). It does NOT raise.

This module imports `data.data_utils` lazily and only for the run
header; if data_utils is not importable (e.g. during the editable
install's own bootstrap, before the package is on the path), the run
header gracefully degrades to a snapshot without the config hashes and
the verification script will reject that run as "no header".

PUBLIC API
----------
    configure(level, format, file_path, ...) -> None
    get_logger(name) -> JsonLogger
    write_run_header(config, seed_state=None) -> None
    reset() -> None                 (test-only; closes and forgets the singleton)
    JsonLogger.debug/info/warning/error/exception(event, **fields)
    JsonLogger.progress(event, **fields)   (throttled on console only)
    JsonLogger.bind(**context) -> JsonLogger  (returns a child with extra context)

See `__all__` at the bottom for the exact surface.
"""

from __future__ import annotations

import datetime as _dt
import importlib.metadata as _ilm
import json
import os
import platform
import socket
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Reserved top-level keys. A caller that passes one of these as a
# keyword argument to a log method gets a TypeError, which is the right
# behaviour: a silent overwrite of `level` by a stray `level=` kwarg
# would corrupt the record in a way that is very hard to debug.
RESERVED_KEYS: frozenset = frozenset({
    "time", "level", "logger", "event", "run_id",
    "pid", "thread", "fields",
    "exc_type", "exc_message", "exc_traceback",
})

# Key patterns that trigger redaction. Matching is by substring on the
# lowercased key, so `api_key`, `API_KEY`, and `my_api_key` all match
# `api_key`. The patterns are deliberately narrow; a false positive
# (redacting a field that wasn't a secret) is much less harmful than a
# false negative.
REDACT_KEYS: frozenset = frozenset({
    "password", "passwd", "secret", "token", "api_key",
    "apikey", "private_key", "authorization", "bearer",
    "credential", "session_key", "cookie",
})

REDACTED_VALUE = "[REDACTED]"

# Default output directory, matching default_e3_config.yaml §4
# (paths.outputs_logs = "outputs/logs").
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_LOG_DIR = _REPO_ROOT / "outputs" / "logs"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "e3doh.log"

# The pinned packages whose versions are captured in the run header.
# This list matches requirements.txt tiers 1-2 (the numerically
# important packages); tiers 3+ are recorded only if the package is
# actually importable, because a reviewer who runs only the toy mode
# may not have installed the protocol-parsing tier.
_HEADER_PACKAGES: tuple = (
    "numpy", "scipy", "pandas", "torch",
    "scikit-learn", "opacus", "dnspython", "h2", "aioquic",
    "cryptography", "scapy", "dask",
)


# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
# The singleton logger and its configuration. The lock protects both;
# `configure()` may be called from any thread, and `get_logger()` is
# called from many threads.

_LOCK = threading.RLock()
_SINGLETON: Optional["JsonLogger"] = None
_HEADER_WRITTEN: bool = False


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LoggerConfigError(Exception):
    """Raised when configure() is called with an invalid argument."""


# ---------------------------------------------------------------------------
# JsonLogger
# ---------------------------------------------------------------------------

class JsonLogger:
    """
    A JSON-structured logger.

    Instances are obtained via get_logger(name); the class is not meant
    to be instantiated directly. A logger is bound to a name (which
    becomes the `logger` field in every record) and may carry additional
    context fields (via bind()) that are merged into every record.

    The logger is thread-safe.
    """

    def __init__(
        self,
        *,
        name: str,
        level: str,
        fmt: str,
        file_path: Optional[Path],
        console: bool,
        progress_interval: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._name = name
        self._level = _level_value(level)
        self._fmt = fmt
        self._file_path = Path(file_path) if file_path else None
        self._console = bool(console)
        self._progress_interval = float(progress_interval)
        self._context: Dict[str, Any] = dict(context or {})
        self._lock = threading.Lock()
        self._file = self._open_file()
        self._last_console_progress: float = 0.0
        self._rich_console = _make_rich_console() if self._console else None

    # -- lifecycle ---------------------------------------------------------

    def _open_file(self):
        """Open the log file (append mode). Returns None on failure; the
        logger then writes to stderr."""
        if self._file_path is None:
            return None
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            f = open(self._file_path, "a", encoding="utf-8", buffering=1)
            # Warn if the file is already past a reasonable size.
            # This is a warning, not an error; the file is not rotated
            # automatically (rotation would break the "the run header is
            # the first line" invariant of the current run).
            if self._file_path.stat().st_size > 100 * 1024 * 1024:
                sys.stderr.write(
                    f"[e3doh.logger] WARNING: log file "
                    f"{self._file_path} is >100 MB; consider rotating.\n"
                )
            return f
        except OSError as e:
            sys.stderr.write(
                f"[e3doh.logger] WARNING: could not open log file "
                f"{self._file_path}: {e}. Falling back to stderr.\n"
            )
            return None

    def close(self) -> None:
        """Close the log file (if open). Idempotent."""
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except OSError:
                    pass
                self._file = None

    # -- public API --------------------------------------------------------

    def debug(self, event: str, **fields: Any) -> None:
        self._log("DEBUG", event, fields, exc=None)

    def info(self, event: str, **fields: Any) -> None:
        self._log("INFO", event, fields, exc=None)

    def warning(self, event: str, **fields: Any) -> None:
        self._log("WARNING", event, fields, exc=None)

    def error(self, event: str, **fields: Any) -> None:
        self._log("ERROR", event, fields, exc=None)

    def exception(self, event: str, **fields: Any) -> None:
        """Log an ERROR record that includes the current exception's
        type, message, and traceback. Must be called from within an
        except block."""
        exc_type, exc_val, exc_tb = sys.exc_info()
        self._log("ERROR", event, fields, exc=(exc_type, exc_val, exc_tb))

    def progress(self, event: str, **fields: Any) -> None:
        """
        Log a progress event. The file handler always writes; the
        console handler throttles to one per `progress_interval` seconds.

        Use this for per-iteration events (epoch start, batch progress)
        that would otherwise flood the console. Do NOT use it for
        important events; a throttled event is a missing event from the
        console's perspective.
        """
        self._log("INFO", event, fields, exc=None, is_progress=True)

    def bind(self, **context: Any) -> "JsonLogger":
        """Return a child logger that merges `context` into every record."""
        child = JsonLogger.__new__(JsonLogger)
        child._name = self._name
        child._level = self._level
        child._fmt = self._fmt
        child._file_path = self._file_path
        child._console = self._console
        child._progress_interval = self._progress_interval
        child._context = {**self._context, **context}
        child._lock = self._lock           # share the parent's write lock
        child._file = self._file           # share the parent's file handle
        child._last_console_progress = 0.0
        child._rich_console = self._rich_console
        return child

    # -- internal ----------------------------------------------------------

    def _log(
        self,
        level: str,
        event: str,
        fields: Dict[str, Any],
        exc: Optional[tuple],
        *,
        is_progress: bool = False,
    ) -> None:
        if _level_value(level) < self._level:
            return

        # Reserved-key check. If a caller passes a reserved key in
        # `fields`, raise immediately. The alternative — silently
        # renaming the key — would make the failure mode very hard to
        # diagnose.
        bad = RESERVED_KEYS.intersection(fields.keys())
        if bad:
            raise TypeError(
                f"JsonLogger.{level.lower()}() received reserved "
                f"keyword argument(s) {sorted(bad)}. "
                f"Reserved keys: {sorted(RESERVED_KEYS)}. "
                f"Rename the field (e.g. `field_event=` instead of `event=`)."
            )

        # Merge context, then redact.
        merged = {**self._context, **fields}
        merged = _redact(merged)

        record = self._build_record(level, event, merged, exc)

        with self._lock:
            self._write_file(record)
            if self._console:
                self._write_console(record, is_progress=is_progress)

    def _build_record(
        self,
        level: str,
        event: str,
        fields: Dict[str, Any],
        exc: Optional[tuple],
    ) -> Dict[str, Any]:
        now = _dt.datetime.now(_dt.timezone.utc)
        record: Dict[str, Any] = {
            "time": now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z",
            "level": level,
            "logger": self._name,
            "event": str(event),
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "fields": fields,
        }
        if exc is not None and exc[0] is not None:
            exc_type, exc_val, exc_tb = exc
            record["exc_type"] = exc_type.__name__
            record["exc_message"] = str(exc_val)
            record["exc_traceback"] = "".join(
                traceback.format_exception(exc_type, exc_val, exc_tb)
            )
        return record

    def _write_file(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        if self._file is not None:
            try:
                self._file.write(line)
                return
            except (OSError, ValueError):
                # Fall through to stderr.
                pass
        sys.stderr.write(line)

    def _write_console(self, record: Dict[str, Any], *, is_progress: bool) -> None:
        # Throttle progress on console only.
        if is_progress and self._progress_interval > 0:
            import time as _t
            now = _t.monotonic()
            if now - self._last_console_progress < self._progress_interval:
                return
            self._last_console_progress = now

        if self._fmt == "text":
            line = self._render_text(record)
        else:
            # JSON on console too, but one line, for a TTY-less capture.
            line = json.dumps(record, ensure_ascii=False, default=str)

        if self._rich_console is not None:
            color = _LEVEL_COLORS.get(record["level"], "white")
            try:
                self._rich_console.print(line, style=color, highlight=False)
                return
            except Exception:
                pass
        sys.stdout.write(line + "\n")

    def _render_text(self, record: Dict[str, Any]) -> str:
        ts = record["time"]
        lvl = record["level"]
        ev = record["event"]
        fields = record["fields"]
        tail = ""
        if fields:
            tail = " " + " ".join(
                f"{k}={_short(v)}" for k, v in sorted(fields.items())
            )
        exc = ""
        if record.get("exc_type"):
            exc = f" exc={record['exc_type']}: {record['exc_message']}"
        return f"{ts} [{lvl:<7}] {ev}{tail}{exc}"


# ---------------------------------------------------------------------------
# Level handling
# ---------------------------------------------------------------------------

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_LEVEL_COLORS = {
    "DEBUG": "dim",
    "INFO": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
}


def _level_value(level: str) -> int:
    try:
        return _LEVELS[level.upper()]
    except KeyError:
        raise LoggerConfigError(
            f"Unknown log level '{level}'. Expected one of {sorted(_LEVELS)}."
        )


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def _redact(obj: Any) -> Any:
    """
    Recursively redact values whose key matches one of REDACT_KEYS.
    Matching is case-insensitive substring on the lowercased key. String
    values are not inspected (only keys); a value that happens to look
    like a secret is not redacted, because we cannot know it is a
    secret, and over-redacting is a nuisance that trains users to ignore
    the redaction.
    """
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if _is_sensitive_key(k):
                out[k] = REDACTED_VALUE
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_redact(x) for x in obj)
    return obj


def _is_sensitive_key(k: Any) -> bool:
    if not isinstance(k, str):
        return False
    lower = k.lower()
    return any(pat in lower for pat in REDACT_KEYS)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _make_rich_console():
    """
    Return a rich.Console if rich is installed, else None. The logger
    falls back to plain print() when rich is not available; the
    dependency is in requirements.txt tier 9 but the artifact should
    work without it.
    """
    try:
        from rich.console import Console  # type: ignore
        return Console(stderr=False, highlight=False, soft_wrap=False)
    except Exception:
        return None


def _short(v: Any, max_len: int = 40) -> str:
    s = repr(v)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


# ---------------------------------------------------------------------------
# Module-level API
# ---------------------------------------------------------------------------

def configure(
    level: str = "INFO",
    fmt: str = "json",
    file_path: Optional[Any] = None,
    *,
    console: bool = True,
    progress_interval: float = 30.0,
) -> None:
    """
    Configure the global logger. Idempotent: safe to call multiple
    times, each call replaces the previous configuration.

    Parameters
    ----------
    level : "DEBUG" | "INFO" | "WARNING" | "ERROR"
    fmt : "json" | "text"
        The format of the *console* output. The file is always JSONL.
    file_path : path-like, optional
        Where to write the log. If None, uses
        outputs/logs/e3doh.log (matching default_e3_config.yaml §4).
    console : bool
        Whether to echo to stdout. Set False in tests to keep output
        clean.
    progress_interval : float
        Minimum seconds between console progress lines. 0 disables
        throttling (every progress event prints).
    """
    global _SINGLETON, _HEADER_WRITTEN
    if fmt not in ("json", "text"):
        raise LoggerConfigError(f"fmt must be 'json' or 'text', got {fmt!r}")
    with _LOCK:
        if _SINGLETON is not None:
            _SINGLETON.close()
        _SINGLETON = JsonLogger(
            name="e3doh", level=level, fmt=fmt,
            file_path=file_path or _DEFAULT_LOG_FILE,
            console=console, progress_interval=progress_interval,
        )
        _HEADER_WRITTEN = False


def get_logger(name: str) -> JsonLogger:
    """
    Return a logger bound to `name`. If the global logger has not been
    configured, it is created with defaults (INFO, json, console on,
    outputs/logs/e3doh.log).

    Callers that want to configure the logger should call configure()
    *before* the first get_logger() call; otherwise the defaults apply
    for the first call and the configure() call rebuilds the logger
    (any records logged between the two calls go to the default path).
    """
    global _SINGLETON
    with _LOCK:
        if _SINGLETON is None:
            configure()
        assert _SINGLETON is not None
        return _SINGLETON.bind(logger_name=name)


def write_run_header(
    config: Optional[Dict[str, Any]] = None,
    seed_state: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write the run header to the current logger's file.

    The header is written once per process; subsequent calls are no-ops.
    This is deliberate: a script that calls get_logger() and then calls
    write_run_header() after loading config should not end up with two
    headers, because the verification script assumes one header per
    process.

    Parameters
    ----------
    config : dict
        The dict returned by data_utils.load_config(). If None, the
        header is written with an empty config and a warning is logged.
    seed_state : dict
        The dict returned by seed.set_global_seed() or
        seed.describe_seed_state(). If None, seed.py is imported and
        describe_seed_state is called with the config's runtime.seed,
        or the header records "seed_state": None if seed.py is not
        importable.
    """
    global _HEADER_WRITTEN
    with _LOCK:
        if _HEADER_WRITTEN:
            return
        if _SINGLETON is None:
            configure()
        assert _SINGLETON is not None
        log = _SINGLETON.bind(logger_name="e3doh.header")

        if config is None:
            log.warning("run_header_without_config",
                        note="write_run_header called with config=None; "
                             "the header will not contain config hashes.")
            config = {}

        # Seed state.
        if seed_state is None:
            seed_state = _try_describe_seed_state(config)

        payload: Dict[str, Any] = {
            "artifact_version": _safe_get(config, "artifact", "version"),
            "framework": _safe_get(config, "artifact", "framework"),
            "chapter": _safe_get(config, "artifact", "chapter"),
            "run_id": config.get("_run_id"),
            "config_hashes": dict(config.get("_hashes", {})),
            "resolved_config": _redact(_strip_private_keys(config)),
            "git_commit": _git_commit(),
            "python": {
                "version": sys.version.split()[0],
                "implementation": platform.python_implementation(),
                "executable": sys.executable,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "hostname": socket.gethostname(),
            },
            "packages": _dependency_versions(),
            "environment": _reproducibility_env(),
            "seed_state": seed_state,
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
        }
        log.info("run_header", **payload)
        _HEADER_WRITTEN = True


def reset() -> None:
    """
    Close and forget the singleton logger. For tests and for scripts
    that need to reconfigure from scratch (e.g. after loading a
    different config). Never called from the artifact's own runtime
    code; if you find yourself wanting to call it there, you probably
    want configure() instead.
    """
    global _SINGLETON, _HEADER_WRITTEN
    with _LOCK:
        if _SINGLETON is not None:
            _SINGLETON.close()
        _SINGLETON = None
        _HEADER_WRITTEN = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_describe_seed_state(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        from src.utils.seed import describe_seed_state  # type: ignore
    except ImportError:
        return None
    seed = 0
    try:
        seed = int(config.get("runtime", {}).get("seed", 0))
    except (TypeError, ValueError):
        pass
    try:
        return describe_seed_state(seed)
    except Exception:
        return None


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _strip_private_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    """Remove keys starting with '_' (they are internal annotations added
    by data_utils.load_config() and would clutter the header)."""
    return {k: v for k, v in d.items() if not (isinstance(k, str) and k.startswith("_"))}


def _git_commit() -> Optional[str]:
    import subprocess
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode("ascii").strip()
    except Exception:
        return None


def _dependency_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for pkg in _HEADER_PACKAGES:
        try:
            out[pkg] = _ilm.version(pkg)
        except _ilm.PackageNotFoundError:
            out[pkg] = None
    return out


def _reproducibility_env() -> Dict[str, Optional[str]]:
    keys = ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "configure",
    "get_logger",
    "write_run_header",
    "reset",
    "JsonLogger",
    "LoggerConfigError",
    "RESERVED_KEYS",
    "REDACT_KEYS",
    "REDACTED_VALUE",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
# Running `python -m src.utils.logger` writes a small log to a temp
# directory and verifies the JSON schema, the redaction, the reserved-
# key check, the level filter, the header, and the text format. It is
# a smoke test; the full test suite (tests/test_logger.py, later in the
# sequence) checks the same behaviours plus the file-rotation and
# concurrent-write paths.

def _self_test() -> int:
    import tempfile
    rc = 0
    print("=== logger self-test ===")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "selftest.log"
        configure(level="DEBUG", fmt="json", file_path=path, console=False)
        log = get_logger("selftest")

        # 1. Basic record.
        log.info("hello", x=1, y="two")
        # 2. Debug record.
        log.debug("dbg", note="should appear at DEBUG")
        # 3. Redaction.
        log.info("creds", password="hunter2", api_key="sk-xxx", safe="ok")
        # 4. Reserved-key check.
        try:
            log.info("bad", level="INFO")
        except TypeError:
            print(f"[OK]   reserved-key check raises")
        else:
            print(f"[FAIL] reserved-key check did not raise")
            rc = 1
        # 5. Exception capture.
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("caught", context="test")
        # 6. Run header.
        write_run_header(config={
            "artifact": {"version": "1.0.0", "framework": "E3-DoH", "chapter": 5},
            "_run_id": "deadbeefcafebabe",
            "_hashes": {"default_e3_config": "a" * 64},
        }, seed_state={"run_seed": 42, "component_seeds": {"aee_training": 42}})

        # 7. Verify file.
        lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        events = [r["event"] for r in lines]
        if "hello" not in events or "dbg" not in events:
            print(f"[FAIL] events: {events}")
            rc = 1
        else:
            print(f"[OK]   {len(lines)} records, events include hello/dbg")
        # Header first.
        if lines[0]["event"] != "run_header":
            print(f"[FAIL] first record is not run_header: {lines[0]['event']}")
            rc = 1
        else:
            print(f"[OK]   run_header is the first record")
        # Redaction.
        creds = next(r for r in lines if r["event"] == "creds")
        if creds["fields"]["password"] != REDACTED_VALUE:
            print(f"[FAIL] password not redacted: {creds['fields']}")
            rc = 1
        elif creds["fields"]["api_key"] != REDACTED_VALUE:
            print(f"[FAIL] api_key not redacted: {creds['fields']}")
            rc = 1
        elif creds["fields"]["safe"] != "ok":
            print(f"[FAIL] non-secret field was redacted: {creds['fields']}")
            rc = 1
        else:
            print(f"[OK]   redaction works (2 secret fields redacted, 1 safe kept)")
        # Exception.
        caught = next(r for r in lines if r["event"] == "caught")
        if caught.get("exc_type") != "ValueError" or "boom" not in caught.get("exc_message", ""):
            print(f"[FAIL] exception record: {caught}")
            rc = 1
        else:
            print(f"[OK]   exception record: {caught['exc_type']}: {caught['exc_message']}")
        # Header content.
        hdr = lines[0]
        if hdr["fields"].get("run_id") != "deadbeefcafebabe":
            print(f"[FAIL] header run_id: {hdr['fields'].get('run_id')}")
            rc = 1
        elif hdr["fields"].get("config_hashes", {}).get("default_e3_config") != "a" * 64:
            print(f"[FAIL] header config_hashes: {hdr['fields'].get('config_hashes')}")
            rc = 1
        else:
            print(f"[OK]   header carries run_id and config hashes")
        # Second header call is a no-op.
        write_run_header(config={"artifact": {"version": "1.0.0"}})
        lines2 = path.read_text(encoding="utf-8").splitlines()
        n_headers = sum(1 for l in lines2 if json.loads(l)["event"] == "run_header")
        if n_headers != 1:
            print(f"[FAIL] {n_headers} headers (expected 1)")
            rc = 1
        else:
            print(f"[OK]   second write_run_header() is a no-op")

        # Text format.
        reset()
        path2 = Path(tmp) / "selftest_text.log"
        configure(level="INFO", fmt="text", file_path=path2, console=False)
        get_logger("selftest2").info("text_event", a=1)
        # File is still JSONL; the text format only affects console.
        tlines = path2.read_text(encoding="utf-8").splitlines()
        if not tlines or json.loads(tlines[0])["event"] != "text_event":
            print(f"[FAIL] text format broke the file JSONL invariant")
            rc = 1
        else:
            print(f"[OK]   text format: file remains JSONL")

    print("=== self-test complete ===")
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
