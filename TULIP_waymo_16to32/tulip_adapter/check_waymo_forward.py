"""TULIP forward-only smoke-test entry point."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--in-chans", type=int, default=2)
    parser.add_argument("--low-height", type=int, default=16)
    parser.add_argument("--high-height", type=int, default=32)
    parser.add_argument("--width", type=int, required=True)
    parser.parse_args()
    raise NotImplementedError("Forward smoke test is intentionally scaffolded only.")


if __name__ == "__main__":
    main()
