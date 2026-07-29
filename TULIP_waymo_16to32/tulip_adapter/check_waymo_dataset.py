"""Dataset-only smoke-test entry point."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.parse_args()
    raise NotImplementedError("Dataset smoke test is intentionally scaffolded only.")


if __name__ == "__main__":
    main()
