from __future__ import annotations

import argparse
import json


def inventory_summary(available: int, requested: int, unit_price: float) -> dict[str, float | int | bool]:
    return {
        "available_units": available,
        "requested_units": requested,
        "can_fulfill": available >= requested,
        "missing_units": max(0, requested - available),
        "estimated_total_eur": round(requested * unit_price, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume disponibilidad y coste estimado.")
    parser.add_argument("--available", type=int, required=True)
    parser.add_argument("--requested", type=int, required=True)
    parser.add_argument("--unit-price", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(inventory_summary(args.available, args.requested, args.unit_price), indent=2))


if __name__ == "__main__":
    main()
