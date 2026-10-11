#!/usr/bin/env python3
"""Add unused source months to an existing run and rebuild its processed dataset."""

import sys

from IT3160.data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["refill", *sys.argv[1:]]))
