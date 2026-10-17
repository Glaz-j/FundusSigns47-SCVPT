#!/usr/bin/env python3
"""Backward-compatible wrapper for building vision1."""

from __future__ import annotations

from build_dataset_version import build_dataset


if __name__ == "__main__":
    build_dataset("vision1")
