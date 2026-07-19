"""Evaluation pipeline launcher script.

Imports and runs the evaluation metrics pipeline.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.evaluation.evaluate import run_evaluation

def main():
    print("Initializing Model Evaluation and Diagnostics Pipeline...")
    try:
        run_evaluation()
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
