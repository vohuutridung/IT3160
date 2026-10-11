#!/usr/bin/env python3
"""Publish a completed local run, including after refill or an upload retry."""

import sys

from IT3160.data.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["upload", *sys.argv[1:]]))
