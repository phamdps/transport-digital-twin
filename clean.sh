#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🧹 Cleaning up unnecessary files and cache directories..."

# Remove all Python cache directories (__pycache__)
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "✔ Removed all __pycache__ directories."

# Remove all compiled Python files (.pyc, .pyo, .pyd)
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" \) -exec rm -f {} +
echo "✔ Removed all compiled Python files."

# Remove pytest cache directories (.pytest_cache)
find . -type d -name ".pytest_cache" -exec rm -rf {} +
echo "✔ Removed all .pytest_cache directories."

# Remove build, dist, and egg-info directories if they exist
rm -rf build/ dist/ *.egg-info .eggs/
echo "✔ Removed build, dist, and packaging artifacts."

echo "✨ Cleanup complete!"