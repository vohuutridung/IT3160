#!/usr/bin/env python3
"""Download prepared HF sources separately, for example extra months for refill."""

import sys

from IT3160.data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download", *sys.argv[1:]]))
