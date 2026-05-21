from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import ensure_data
from src.recommender import HybridBookRecommender


ARTIFACT_PATH = Path("artifacts/recommender.joblib")


def main() -> None:
    books, ratings = ensure_data()
    model = HybridBookRecommender()
    model.fit(books, ratings)
    model.save(ARTIFACT_PATH)
    print(f"Trained on {len(books)} books and {len(ratings)} reader interactions.")
    print(f"Saved model to {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
