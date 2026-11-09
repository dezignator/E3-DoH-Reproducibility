#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/reproduce_figures/figure_11_architecture.py
====================================================

E3-DoH Reproducibility — Figure 11 reproduction (architecture diagram).

Chapter 5 of the thesis: E3-DoH (Enhanced Evolutionary Encryption for
Multi-Protocol DNS Security — DoH, DoT, DoQ).

Figure 11 (p.118) is the E3-DoH architecture diagram. It shows the
data plane (traffic ingestion, feature extraction, adaptive
evolutionary encryption, quantum-safe homomorphic encryption,
anomaly detection, adaptive multi-protocol sandboxing) with three
protocol-specific input streams (DoH, DoT, DoQ) flowing through the
common abstraction layer into the shared defense stack.

The figure is a *diagram*, not a data plot. It has no underlying
numerical data. It is regenerated from a fixed layout specification
(box positions, labels, arrows) so that a fresh run produces the
same SVG bytes as the committed reference.

WHAT THE SCRIPT DOES
--------------------
    - Draws the diagram from a fixed layout spec (below).
    - Writes an SVG to outputs/figures/figure_11.svg.
    - Optionally verifies against the committed SVG in
      expected_outputs/figure_11_expected.svg.

Because the figure has no numerical data, the verification is a
byte-for-byte comparison (after normalising whitespace and timestamp).
The output is deliberately structured so that a reviewer can diff
the generated SVG against the committed one and see exactly what
changed if anything does.

USAGE
-----
    python scripts/reproduce_figures/figure_11_architecture.py
    python scripts/reproduce_figures/figure_11_architecture.py --verify
    python scripts/reproduce_figures/figure_11_architecture.py --commit
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.data_utils import (
    ConfigError,
    load_config,
    output_path,
    expected_path,
)
from src.utils.logger import configure as configure_logger
from src.utils.logger import get_logger


# ---------------------------------------------------------------------------
# The layout specification
# ---------------------------------------------------------------------------
# Every element of the diagram is a box, an arrow, or a text label,
# with an explicit (x, y, width, height) and a text content. The spec
# is the *only* input to the SVG generator; there is no random
# element, no data-driven variation, and no font-dependent sizing.
#
# Boxes are drawn with rounded rectangles; arrows are drawn as
# straight lines with arrowheads. Both are pure SVG primitives
# (rect, line, path) that do not depend on the renderer's font.

# The canvas dimensions.
_CANVAS_WIDTH = 1200
_CANVAS_HEIGHT = 900

# Colour palette. Colours are hex strings so the SVG is
# self-contained (no external CSS).
_COLORS = {
    "input_doh": "#4A90D9",       # blue
    "input_dot": "#4A90D9",
    "input_doq": "#4A90D9",
    "abstraction": "#7B68EE",     # purple
    "core_aee": "#50A050",        # green
    "core_qhe": "#50A050",
    "detection": "#E8A020",       # amber
    "sandbox": "#E85D5D",         # red
    "mitigation": "#5D5D5D",      # grey
    "background": "#FFFFFF",
    "text": "#1A1A1A",
    "text_light": "#FFFFFF",
    "arrow": "#333333",
    "arrow_label": "#666666",
    "border": "#333333",
}

# The boxes: name -> (x, y, w, h, label, colour_key).
# The layout is a top-down flow:
#   1. Three input streams at the top (DoH, DoT, DoQ).
#   2. A common abstraction layer below them.
#   3. The core defense stack below that (AEE, QHE, detection).
#   4. The sandbox and mitigation at the bottom.
_BOXES: Dict[str, Tuple[float, float, float, float, str, str]] = {
    # Row 1: input streams.
    "input_doh": (100, 40, 200, 70, "DoH Stream\n(TCP/443, TLS 1.3)", "input_doh"),
    "input_dot": (500, 40, 200, 70, "DoT Stream\n(TCP/853, TLS 1.3)", "input_dot"),
    "input_doq": (900, 40, 200, 70, "DoQ Stream\n(UDP/853, QUIC v1)", "input_doq"),

    # Row 2: abstraction layer.
    "abstraction": (200, 180, 800, 70,
                     "Multi-Protocol Abstraction Layer\n(common metadata representation)",
                     "abstraction"),

    # Row 3: core defense stack (three columns).
    "aee": (100, 330, 300, 120, "Adaptive Evolutionary Encryption\nModule-LWE + entropy-driven key schedule",
            "core_aee"),
    "qhe": (450, 330, 300, 120, "Quantum-Safe Homomorphic Encryption\nRing-LWE + selective operation set",
            "core_qhe"),
    "detection": (800, 330, 300, 120, "Anomaly Detection\nRiemannian geodesic flow residual",
                   "detection"),

    # Row 4: sandbox and mitigation (spanning).
    "sandbox": (100, 530, 500, 100, "Adaptive Multi-Protocol Sandboxing (AMPS)\nper-flow isolation (DoH stream / DoT TLS / DoQ connection migration)",
                 "sandbox"),
    "mitigation": (650, 530, 450, 100, "Mitigation + Reversible Actions\nrate limit / alert / key rotation / traffic shaping",
                    "mitigation"),

    # Row 5: output.
    "output": (350, 720, 500, 60, "Protected DNS Resolution\n(benign flows continue; anomalous flows isolated)",
                "background"),
}

# The arrows: (from_box, to_box, label, style). Style is "solid" or
# "dashed". Labels may be empty strings.
_ARROWS: List[Tuple[str, str, str, str]] = [
    ("input_doh", "abstraction", "", "solid"),
    ("input_dot", "abstraction", "", "solid"),
    ("input_doq", "abstraction", "", "solid"),
    ("abstraction", "aee", "", "solid"),
    ("abstraction", "qhe", "", "solid"),
    ("abstraction", "detection", "", "solid"),
    ("aee", "sandbox", "", "solid"),
    ("qhe", "detection", "", "solid"),
    ("detection", "sandbox", "", "solid"),
    ("detection", "mitigation", "anomaly signal", "solid"),
    ("sandbox", "mitigation", "isolation action", "solid"),
    ("sandbox", "output", "", "dashed"),
    ("mitigation", "output", "", "solid"),
]


# ---------------------------------------------------------------------------
# SVG generation
# ---------------------------------------------------------------------------

def _box_center(box: Tuple[float, float, float, float, str, str]) -> Tuple[float, float]:
    x, y, w, h, _, _ = box
    return x + w / 2, y + h / 2


def _edge_points(
    from_box: Tuple[float, float, float, float, str, str],
    to_box: Tuple[float, float, float, float, str, str],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute the arrow's start and end points on the box edges.

    The arrow starts at the bottom-center of the from-box and ends at
    the top-center of the to-box. This gives a clean top-down flow
    with no arrow crossing inside a box.
    """
    fx, fy, fw, fh, _, _ = from_box
    tx, ty, tw, th, _, _ = to_box
    start = (fx + fw / 2, fy + fh)
    end = (tx + tw / 2, ty)
    return start, end


def _render_text_lines(
    x: float, y: float, text: str, color: str, font_size: int = 13,
    font_family: str = "Helvetica, Arial, sans-serif",
) -> List[str]:
    """Split a multi-line label into one <text> per line."""
    lines = []
    for i, line in enumerate(text.split("\n")):
        baseline = y + i * (font_size + 2)
        lines.append(
            f'  <text x="{x:.1f}" y="{baseline:.1f}" '
            f'font-family="{font_family}" font-size="{font_size}" '
            f'fill="{color}" text-anchor="middle">{_escape_xml(line)}</text>'
        )
    return lines


def _escape_xml(s: str) -> str:
    """Escape the XML special characters."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def _render_svg() -> str:
    """
    Render the architecture diagram as an SVG string.

    Every element is drawn from the _BOXES and _ARROWS specs. There is
    no randomness, no data-driven variation, and no font-dependent
    sizing, so the output is deterministic given the spec.
    """
    parts: List[str] = []
    parts.append(f'<?xml version="1.0" encoding="UTF-8"?>')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_CANVAS_WIDTH}" height="{_CANVAS_HEIGHT}" '
        f'viewBox="0 0 {_CANVAS_WIDTH} {_CANVAS_HEIGHT}">'
    )
    parts.append("  <!-- Background -->")
    parts.append(
        f'  <rect x="0" y="0" width="{_CANVAS_WIDTH}" '
        f'height="{_CANVAS_HEIGHT}" fill="{_COLORS["background"]}"/>'
    )
    parts.append(f'  <title>Figure 11: E3-DoH Architecture</title>')
    parts.append(f'  <desc>Architecture diagram generated from a fixed layout spec (deterministic).</desc>')

    # Draw arrows first (so they appear behind boxes).
    parts.append("  <!-- Arrows -->")
    parts.append('  <defs>')
    parts.append(
        '    <marker id="arrowhead" markerWidth="10" markerHeight="7" '
        'refX="9" refY="3.5" orient="auto" markerUnits="strokeWidth">'
    )
    parts.append(
        f'      <polygon points="0 0, 10 3.5, 0 7" fill="{_COLORS["arrow"]}"/>'
    )
    parts.append('    </marker>')
    parts.append('  </defs>')

    for from_name, to_name, label, style in _ARROWS:
        from_box = _BOXES[from_name]
        to_box = _BOXES[to_name]
        start, end = _edge_points(from_box, to_box)
        dash = ' stroke-dasharray="6,4"' if style == "dashed" else ""
        parts.append(
            f'  <line x1="{start[0]:.1f}" y1="{start[1]:.1f}" '
            f'x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
            f'stroke="{_COLORS["arrow"]}" stroke-width="1.5"{dash} '
            f'marker-end="url(#arrowhead)"/>'
        )
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            # Offset the label to the right of the arrow.
            parts.append(
                f'  <text x="{mid_x + 5:.1f}" y="{mid_y - 3:.1f}" '
                f'font-family="Helvetica, Arial, sans-serif" font-size="11" '
                f'fill="{_COLORS["arrow_label"]}" font-style="italic">'
                f'{_escape_xml(label)}</text>'
            )

    # Draw boxes.
    parts.append("  <!-- Boxes -->")
    for name, (x, y, w, h, label, color_key) in _BOXES.items():
        fill = _COLORS[color_key]
        text_color = (
            _COLORS["text"] if color_key == "background" else _COLORS["text_light"]
        )
        # Rounded rectangle.
        parts.append(
            f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="8" ry="8" fill="{fill}" stroke="{_COLORS["border"]}" '
            f'stroke-width="1.5"/>'
        )
        # Label, centered vertically. Multi-line labels are split.
        n_lines = len(label.split("\n"))
        total_h = n_lines * (13 + 2)
        text_y = y + h / 2 - total_h / 2 + 13
        parts.extend(_render_text_lines(
            x + w / 2, text_y, label, text_color, font_size=13,
        ))

    # Title and caption.
    parts.append("  <!-- Title -->")
    parts.append(
        f'  <text x="{_CANVAS_WIDTH / 2}" y="20" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="16" '
        f'font-weight="bold" fill="{_COLORS["text"]}" text-anchor="middle">'
        f'Figure 11: E3-DoH Architecture</text>'
    )
    parts.append(
        f'  <text x="{_CANVAS_WIDTH / 2}" y="{_CANVAS_HEIGHT - 20}" '
        f'font-family="Helvetica, Arial, sans-serif" font-size="11" '
        f'font-style="italic" fill="{_COLORS["text"]}" text-anchor="middle">'
        f'Generated deterministically from a fixed layout spec; '
        f'no data-driven variation.</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _normalise_svg(svg: str) -> str:
    """
    Normalise an SVG for byte-comparison.

    The output is already deterministic, but normalisation removes
    trailing whitespace and collapses any runs of spaces to a single
    space, so that a minor formatting difference (e.g. an editor's
    trailing newline) does not cause a spurious mismatch.
    """
    lines = [line.rstrip() for line in svg.splitlines()]
    # Drop the last empty line if any.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def _verify(generated: str, reference_path: Path, log: Any) -> Dict[str, Any]:
    """Compare the generated SVG against the reference."""
    if not reference_path.exists():
        log.warning("reference_not_found", path=str(reference_path))
        return {"verified": False, "reason": "reference file missing"}
    reference = reference_path.read_text(encoding="utf-8")
    if _normalise_svg(generated) == _normalise_svg(reference):
        return {"verified": True, "n_mismatches": 0}
    # Compute a per-line diff.
    gen_lines = _normalise_svg(generated).splitlines()
    ref_lines = _normalise_svg(reference).splitlines()
    mismatches = []
    for i in range(max(len(gen_lines), len(ref_lines))):
        g = gen_lines[i] if i < len(gen_lines) else "<missing>"
        r = ref_lines[i] if i < len(ref_lines) else "<missing>"
        if g != r:
            mismatches.append({"line": i + 1, "generated": g, "reference": r})
    return {
        "verified": False,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="figure_11_architecture.py",
        description="Reproduce Figure 11 (E3-DoH architecture diagram).",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--commit", action="store_true",
                        help="Write the reference SVG to expected_outputs/ (sign-off only).")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)

    started = time.perf_counter()

    try:
        config = load_config(config_path=args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2
    log_cfg = config.get("logging", {}) or {}
    configure_logger(
        level=args.log_level,
        fmt=str(log_cfg.get("format", "json")),
        file_path=log_cfg.get("file", "outputs/logs/figure_11.log"),
        console=True,
    )
    log = get_logger("e3doh.figures.figure_11")
    log.info("figure_11_start")

    # Render.
    svg = _render_svg()
    log.info("svg_rendered", size_bytes=len(svg),
             n_boxes=len(_BOXES), n_arrows=len(_ARROWS))

    # Write.
    out_path = output_path("figures", "figure_11.svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(svg, encoding="utf-8")
    log.info("svg_written", path=str(out_path))

    if args.commit:
        commit_path = expected_path("figure_11_expected.svg")
        commit_path.parent.mkdir(parents=True, exist_ok=True)
        commit_path.write_text(svg, encoding="utf-8")
        log.info("svg_committed", path=str(commit_path))

    verification = None
    if args.verify:
        verification = _verify(svg, expected_path("figure_11_expected.svg"), log)
        log.info("figure_11_verified", **verification)

    elapsed = round(time.perf_counter() - started, 3)
    print()
    print(f"=== Figure 11 (architecture) ===")
    print(f"Boxes: {len(_BOXES)}, Arrows: {len(_ARROWS)}")
    print(f"SVG size: {len(svg)} bytes")
    print(f"Output: {out_path}")
    if verification is not None:
        if verification["verified"]:
            print(f"Verification: PASS (byte-for-byte match)")
        else:
            print(f"Verification: FAIL ({verification.get('n_mismatches', '?')} line mismatches)")
            for m in verification.get("mismatches", [])[:5]:
                print(f"  line {m['line']}:")
                print(f"    generated: {m['generated']}")
                print(f"    reference: {m['reference']}")
    print(f"Wall time: {elapsed:.3f} s")
    print()

    if verification is not None and not verification["verified"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
