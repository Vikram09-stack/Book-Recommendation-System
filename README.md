# Book Recommendation System

A complete Streamlit book recommendation website powered by a trained machine learning recommender and live Open Library cover data.

The model combines:

- Supervised model selection across Ridge Regression, KNN Regressor, and Random Forest Regressor using validation RMSE.
- Logistic Regression as a like/dislike baseline.
- Content-based retrieval with TF-IDF over titles, authors, genres, moods, themes, and descriptions.
- Collaborative filtering with truncated SVD over reader/book interaction data.
- Open Library Search API integration directly inside the recommendation page for live book metadata and cover art.
- User controls for genre, mood, book length, difficulty, and recommendation strategy.

## Run It

```powershell
python -m pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

The training script writes model artifacts to `artifacts/recommender.joblib`.

## Project Structure

- `app.py` - Streamlit landing page and recommendation website.
- `src/data.py` - Curated starter catalog and interaction generation.
- `src/recommender.py` - Supervised hybrid recommender model.
- `src/open_library.py` - Open Library API and cover helpers.
- `src/train_model.py` - End-to-end training script.
- `data/` - Generated CSV datasets.
- `artifacts/` - Trained model artifacts.

## Notes

The included catalog is a high-quality local starter dataset so the app works offline. To scale it, replace `data/books.csv` and `data/ratings.csv` with a larger dataset such as Book-Crossing, Goodreads exports, or your own bookstore/library interactions, then rerun training.
