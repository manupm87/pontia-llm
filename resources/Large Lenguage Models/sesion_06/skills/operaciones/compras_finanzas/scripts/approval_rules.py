from __future__ import annotations

import argparse


def approval_path(amount_eur: float) -> str:
    if amount_eur < 500:
        return "manager_directo"
    if amount_eur <= 2_000:
        return "responsable_area_y_finanzas"
    return "direccion_y_orden_de_compra"


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula el circuito de aprobacion por importe.")
    parser.add_argument("amount_eur", type=float)
    args = parser.parse_args()
    print(approval_path(args.amount_eur))


if __name__ == "__main__":
    main()
