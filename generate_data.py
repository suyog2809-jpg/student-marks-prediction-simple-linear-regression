"""Synthetic Educational Dataset Generator for Student Marks Prediction System.

This script generates synthetic, realistic educational demonstration data
modeling the relationship between daily study hours and final examination marks.

DISCLOSURE AND TRANSPARENCY STATEMENT:
--------------------------------------
All records produced by this script are entirely SYNTHETIC and generated
strictly for academic demonstration and machine learning training purposes.
They DO NOT represent actual student performance, real academic institutions,
or verified empirical records. Academic performance in the real world is
multifactorial (attendance, health, prior foundation, teaching quality, etc.)
and cannot be determined solely by study hours.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = ROOT / "student_marks_dataset.csv"

# Empirical constants for the synthetic data generation
# Equation: Marks = Intercept (18.0) + Slope (7.1) * Hours + Gaussian Noise (mean 0, std 8.0)
DEFAULT_NUM_RECORDS = 3200  # Within required 3,000–5,000 range
DEFAULT_RANDOM_SEED = 42
MIN_HOURS = 0.5
MAX_HOURS = 10.0


def generate_dataset(
    n_samples: int = DEFAULT_NUM_RECORDS,
    random_seed: int = DEFAULT_RANDOM_SEED,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Generate synthetic records modeling study hours vs examination marks.

    Args:
        n_samples: Total number of student records to create (default: 3200).
        random_seed: Seed for NumPy random number generator for reproducibility.
        output_path: Destination filepath for the CSV dataset.

    Returns:
        pd.DataFrame containing columns: Student_ID, Study_Hours, Final_Marks.
    """
    rng = np.random.default_rng(random_seed)

    # Study hours sampled uniformly across realistic daily range [0.5, 10.0]
    hours = rng.uniform(MIN_HOURS, MAX_HOURS, n_samples)

    # True underlying linear relationship with realistic educational noise
    # Intercept = 18 marks (baseline score), Slope = 7.1 marks/hour
    # Noise standard deviation = 8.0 marks (captures non-linear student variation)
    noise = rng.normal(loc=0.0, scale=8.0, size=n_samples)
    marks_raw = 18.0 + (7.1 * hours) + noise

    # Strict physical boundary clipping for academic examination marks [0, 100]
    marks_clipped = np.clip(marks_raw, 0.0, 100.0)

    # Format into professional structured DataFrame
    df = pd.DataFrame({
        "Student_ID": [f"STU{i:04d}" for i in range(1, n_samples + 1)],
        "Study_Hours": np.round(hours, 2),
        "Final_Marks": np.round(marks_clipped, 2),
    })

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic student marks dataset for Simple Linear Regression demonstration."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_NUM_RECORDS,
        help="Number of student records to generate (default: 3200, recommended: 3000-5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for deterministic reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output CSV destination path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 70)
    print("STUDENT MARKS DATASET GENERATOR (ACADEMIC SYNTHETIC DEMONSTRATION)")
    print("=" * 70)
    print(f"Target record count : {args.samples:,}")
    print(f"Random seed         : {args.seed}")
    print(f"Output path         : {args.output}")

    dataset = generate_dataset(
        n_samples=args.samples,
        random_seed=args.seed,
        output_path=Path(args.output),
    )

    print("-" * 70)
    print(f"Generation successful!")
    print(f"Total records       : {len(dataset):,}")
    print(f"Study Hours range   : {dataset['Study_Hours'].min():.2f} to {dataset['Study_Hours'].max():.2f} hrs/day")
    print(f"Final Marks range   : {dataset['Final_Marks'].min():.2f} to {dataset['Final_Marks'].max():.2f} / 100")
    print(f"Mean Study Hours    : {dataset['Study_Hours'].mean():.2f} hrs")
    print(f"Mean Final Marks    : {dataset['Final_Marks'].mean():.2f} marks")
    print(f"File saved at       : {Path(args.output).resolve()}")
    print("=" * 70)
