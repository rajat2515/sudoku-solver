#!/usr/bin/env python3
"""
Backward-compatibility entry point.
Redirects to modern app.py CLI interface.
"""

from app import run

if __name__ == "__main__":
    run()
