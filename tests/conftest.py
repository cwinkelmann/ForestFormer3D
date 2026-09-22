"""Shared pytest setup: put the checkout root on sys.path so tests can import repo packages.

Inside the Docker image PYTHONPATH=/workspace already does this; on the Mac it lets
Phase 1's CPU tests import pure-Python modules by file path.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
