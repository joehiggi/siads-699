#!/usr/bin/env python3
"""Estimate how many labeled images are needed to fine-tune YOLO detections."""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist
from typing import Dict


def parse_class_boxes(raw: str) -> Dict[str, float]:
    """Parse class:boxes_per_image pairs (e.g., header:1,body:1,footer:1)."""
    entries = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ValueError(f"Expected name:value pair, got '{chunk}'")
        name, value = chunk.split(":", maxsplit=1)
        entries[name.strip()] = float(value)
    if not entries:
        raise ValueError("At least one class:value pair is required.")
    return entries


def z_score(confidence: float) -> float:
    """Return the z-score for a (0,1) confidence level."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def required_boxes(z: float, margin: float, base_rate: float) -> float:
    """n = (z^2 * p * (1 - p)) / E^2."""
    if not 0 < margin < 1:
        raise ValueError("margin must be in (0, 1)")
    if not 0 < base_rate < 1:
        raise ValueError("base_rate must be in (0, 1)")
    return (z**2 * base_rate * (1.0 - base_rate)) / (margin**2)


def margin_from_boxes(z: float, base_rate: float, boxes: float) -> float:
    if boxes <= 0:
        raise ValueError("boxes must be positive")
    return z * math.sqrt(base_rate * (1.0 - base_rate) / boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the recommended number of labeled images for YOLO fine-tuning "
            "using the standard proportion sample size equation."
        )
    )
    parser.add_argument(
        "--class-boxes",
        default="header:1,body:1,footer:1",
        help=(
            "Comma-separated class:boxes_per_image pairs. "
            "Use values >=1 when multiple instances of a class appear per image."
        ),
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level (default: 0.95).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Desired margin of error (fraction). Default 0.05 == ±5%%.",
    )
    parser.add_argument(
        "--base-rate",
        type=float,
        default=0.5,
        help=(
            "Assumed per-class detection rate when estimating uncertainty. "
            "Use 0.5 for the most conservative (largest) sample size."
        ),
    )
    parser.add_argument(
        "--current-images",
        type=int,
        default=None,
        help="Optional. Report the worst-case margin achieved by N labeled images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = parse_class_boxes(args.class_boxes)
    confidence = args.confidence
    if confidence > 1:
        if confidence >= 100:
            raise SystemExit("confidence percentage must be less than 100.")
        confidence /= 100.0
    z = z_score(confidence)
    boxes_needed = required_boxes(z, args.margin, args.base_rate)

    per_class: Dict[str, int] = {}
    for name, boxes_per_image in classes.items():
        if boxes_per_image <= 0:
            raise SystemExit(f"boxes_per_image for {name} must be positive.")
        per_class[name] = math.ceil(boxes_needed / boxes_per_image)

    recommended = max(per_class.values())
    print(
        f"Target margin ±{args.margin * 100:.1f}% at {confidence * 100:.1f}% "
        f"confidence (z={z:.2f}) with base rate {args.base_rate:.2f}"
    )
    print(f"Required boxes per class: {math.ceil(boxes_needed)}")
    print("Images needed per class:")
    for name, count in per_class.items():
        print(f"  - {name}: {count}")
    print(f"Recommended labeled images: {recommended}")

    if args.current_images:
        print()
        print(f"With {args.current_images} labeled images:")
        for name, boxes_per_image in classes.items():
            boxes = args.current_images * boxes_per_image
            margin = margin_from_boxes(z, args.base_rate, boxes)
            print(
                f"  - {name}: worst-case margin ±{margin * 100:.1f}% "
                f"({boxes} labeled boxes)"
            )


if __name__ == "__main__":
    main()
