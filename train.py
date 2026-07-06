"""
UCOD-ADF Training Entry Point.

Usage:
    python train.py --config configs/uscod/ADF_dinov2.py
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.train import main

if __name__ == '__main__':
    main()
