"""Pytest configuration and headless backend setup."""

import matplotlib

# Force headless Agg backend for all automated test runs
matplotlib.use("Agg")
