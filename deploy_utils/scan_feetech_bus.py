"""Scan a Feetech serial bus for visible motor IDs."""

import argparse

from lerobot.motors.feetech import FeetechMotorsBus


def main():
    parser = argparse.ArgumentParser(description="Scan visible Feetech motor IDs")
    parser.add_argument("--port", default="/dev/ttyACM0")
    args = parser.parse_args()

    found = FeetechMotorsBus.scan_port(args.port)
    if not found:
        print(f"No motors found on {args.port}")
        return

    print(f"Motors found on {args.port}:")
    for baudrate, ids in sorted(found.items()):
        print(f"  baudrate {baudrate}: {ids}")


if __name__ == "__main__":
    main()
