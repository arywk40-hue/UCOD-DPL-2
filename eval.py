"""
UCOD-ADF Evaluation Entry Point.

Usage:
    python eval.py --config configs/uscod/ADF_dinov2.py --load_from work_dirs/uscod/ADF_dinov2/latest.pth
"""

import os
import sys

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.eval import main

if __name__ == '__main__':
    main()
