from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler


PACE_ORDER = {"slow": 0, "medium": 1, "fast": 2}
LENGTH_ORDER = {"short": 0, "medium": 1, "long": 2}
DIFFICULTY_ORDER = {"low": 0, "medium": 1, "high": 2}


@dataclass
class RecommendationResult:
    book_id: int
    score: float
    title: str
    author: str
    year: int
    genres: str
    moods: str
    pace: str
    length: str
    difficulty: str
    description: str
    explanation: str


class HybridBookRecommender:
    def __init__(self, content_weight: float = 0.58, collaborative_weight: float = 0.32, popularity_weight: float = 0.10) -> None:
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.popularity_weight = popularity_weight
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=32, random_state=42)
        self.scaler = MinMaxScaler()
        self.supervised_model = None
        self.model_report: dict[str, object] = {}

    def fit(self, books: pd.DataFrame, ratings: pd.DataFrame) -> "HybridBookRecommender":
        self.books = books.copy().reset_index(drop=True)
        self.ratings = ratings.copy()
        self.book_ids = self.books["book_id"].astype(int).to_numpy()
        self.book_id_to_index = {int(book_id): index for index, book_id in enumerate(self.book_ids)}

        text_corpus = self.books.apply(self._build_text, axis=1)
        self.content_matrix = self.vectorizer.fit_transform(text_corpus)
        self.content_similarity = cosine_similarity(self.content_matrix)

        rating_matrix = self._build_rating_matrix()
        n_components = max(2, min(32, min(rating_matrix.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.item_factors = self.svd.fit_transform(rating_matrix.T)
        self.collaborative_similarity = cosine_similarity(self.item_factors)

        popularity = (
            self.ratings.groupby("book_id")["rating"]
            .agg(["mean", "count"])
            .reindex(self.book_ids)
            .fillna({"mean": self.ratings["rating"].mean(), "count": 0})
        )
        global_mean = float(self.ratings["rating"].mean())
        smoothing = 25
        weighted_rating = (popularity["count"] * popularity["mean"] + smoothing * global_mean) / (popularity["count"] + smoothing)
        scaled = self.scaler.fit_transform(weighted_rating.to_numpy().reshape(-1, 1)).ravel()
        self.popularity_score = scaled
        self.book_stats = popularity.rename(columns={"mean": "book_mean", "count": "book_count"})
        self.book_stats["book_count"] = self.book_stats["book_count"].astype(float)
        self.global_mean = global_mean
        self._train_supervised_ranker()
        return self

    def recommend(
        self,
        liked_book_ids: Iterable[int] | None = None,
        query: str = "",
        genres: Iterable[str] | None = None,
        moods: Iterable[str] | None = None,
        pace: str | None = None,
        length: str | None = None,
        difficulty: str | None = None,
        strategy: str = "Balanced",
        top_n: int = 12,
    ) -> list[RecommendationResult]:
        liked = [int(book_id) for book_id in (liked_book_ids or []) if int(book_id) in self.book_id_to_index]
        selected_indexes = [self.book_id_to_index[book_id] for book_id in liked]
        query = query.strip()
        feature_frame = self._build_inference_features(liked, query, genres, moods, pace, length, difficulty)
        if self.supervised_model is not None:
            candidate_scores = self.supervised_model.predict(feature_frame[self.feature_columns])
            candidate_scores = np.asarray(candidate_scores, dtype=float) / 5.0
        else:
            candidate_scores = np.zeros(len(self.books), dtype=float)

        if query:
            query_vector = self.vectorizer.transform([query])
            candidate_scores += 0.42 * cosine_similarity(query_vector, self.content_matrix).ravel()

        content_weight, collaborative_weight, popularity_weight = self._strategy_weights(strategy)
        if selected_indexes:
            candidate_scores += 0.16 * content_weight * self.content_similarity[selected_indexes].mean(axis=0)
            candidate_scores += 0.18 * collaborative_weight * self.collaborative_similarity[selected_indexes].mean(axis=0)
        else:
            candidate_scores += popularity_weight * self.popularity_score

        candidate_scores += popularity_weight * self.popularity_score
        candidate_scores += self._preference_bonus(genres, moods, pace, length, difficulty)

        if selected_indexes:
            candidate_scores[selected_indexes] = -np.inf

        ranked_indexes = np.argsort(candidate_scores)[::-1][:top_n]
        return [self._format_result(index, candidate_scores[index], liked, genres, moods, pace, length, difficulty) for index in ranked_indexes]

    def similar_books(self, book_id: int, top_n: int = 8) -> list[RecommendationResult]:
        if int(book_id) not in self.book_id_to_index:
            return []
        index = self.book_id_to_index[int(book_id)]
        scores = (
            self.content_weight * self.content_similarity[index]
            + self.collaborative_weight * self.collaborative_similarity[index]
            + self.popularity_weight * self.popularity_score
        )
        scores[index] = -np.inf
        ranked = np.argsort(scores)[::-1][:top_n]
        return [self._format_result(book_index, scores[book_index], [book_id], None, None, None, None, None) for book_index in ranked]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "HybridBookRecommender":
        return joblib.load(path)

    def _build_text(self, row: pd.Series) -> str:
        return " ".join(
            [
                str(row["title"]),
                str(row["author"]),
                str(row["genres"]).replace(";", " "),
                str(row["moods"]).replace(";", " "),
                str(row["pace"]),
                str(row["length"]),
                str(row["difficulty"]),
                str(row["description"]),
            ]
        )

    def _build_rating_matrix(self) -> np.ndarray:
        users = sorted(self.ratings["user_id"].unique())
        user_to_index = {user_id: index for index, user_id in enumerate(users)}
        matrix = np.zeros((len(users), len(self.books)), dtype=float)
        for row in self.ratings.itertuples(index=False):
            matrix[user_to_index[row.user_id], self.book_id_to_index[int(row.book_id)]] = float(row.rating)
        user_means = np.true_divide(matrix.sum(axis=1), (matrix > 0).sum(axis=1), where=(matrix > 0).sum(axis=1) != 0)
        matrix = np.where(matrix > 0, matrix - user_means[:, None], 0)
        return matrix

    def _train_supervised_ranker(self) -> None:
        feature_frame = self._build_training_features()
        target = feature_frame.pop("rating")
        train_x, test_x, train_y, test_y = train_test_split(
            feature_frame,
            target,
            test_size=0.22,
            random_state=42,
            stratify=(target >= 4).astype(int),
        )

        models = {
            "Ridge regression": Ridge(alpha=1.6),
            "KNN regressor": KNeighborsRegressor(n_neighbors=18, weights="distance"),
            "Random forest regressor": RandomForestRegressor(
                n_estimators=260,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        }
        report = {}
        best_name = ""
        best_rmse = float("inf")
        best_model = None
        for name, model in models.items():
            model.fit(train_x, train_y)
            predictions = np.clip(model.predict(test_x), 1, 5)
            rmse = float(np.sqrt(mean_squared_error(test_y, predictions)))
            report[name] = {
                "rmse": rmse,
                "mae": float(mean_absolute_error(test_y, predictions)),
            }
            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_model = model

        classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
        classifier.fit(train_x, (train_y >= 4).astype(int))
        like_predictions = classifier.predict(test_x)
        report["Logistic regression classifier"] = {
            "accuracy": float(accuracy_score((test_y >= 4).astype(int), like_predictions)),
            "target": "predicts whether a reader will rate a book 4+ stars",
        }

        self.supervised_model = best_model
        self.feature_columns = list(train_x.columns)
        self.model_report = {
            "selected_model": best_name,
            "selection_metric": "lowest validation RMSE",
            "validation": report,
        }

    def _build_training_features(self) -> pd.DataFrame:
        high_ratings = self.ratings[self.ratings["rating"] >= 4]
        liked_by_user = high_ratings.groupby("user_id")["book_id"].apply(list).to_dict()
        user_stats = self.ratings.groupby("user_id")["rating"].agg(["mean", "count"]).rename(columns={"mean": "user_mean", "count": "user_count"})
        rows = []
        for row in self.ratings.itertuples(index=False):
            book_index = self.book_id_to_index[int(row.book_id)]
            liked_ids = [book_id for book_id in liked_by_user.get(row.user_id, []) if int(book_id) != int(row.book_id)]
            liked_indexes = [self.book_id_to_index[int(book_id)] for book_id in liked_ids if int(book_id) in self.book_id_to_index]
            user_row = user_stats.loc[row.user_id]
            rows.append(
                {
                    **self._book_feature_dict(book_index),
                    "user_mean": float(user_row["user_mean"]),
                    "user_count": float(user_row["user_count"]),
                    "genre_affinity": self._term_affinity(book_index, liked_indexes, "genres"),
                    "mood_affinity": self._term_affinity(book_index, liked_indexes, "moods"),
                    "author_affinity": self._author_affinity(book_index, liked_indexes),
                    "content_profile_similarity": self._average_similarity(book_index, liked_indexes, self.content_similarity),
                    "collaborative_profile_similarity": self._average_similarity(book_index, liked_indexes, self.collaborative_similarity),
                    "query_similarity": 0.0,
                    "preference_match": 0.0,
                    "rating": float(row.rating),
                }
            )
        return pd.DataFrame(rows)

    def _build_inference_features(
        self,
        liked_book_ids: list[int],
        query: str,
        genres: Iterable[str] | None,
        moods: Iterable[str] | None,
        pace: str | None,
        length: str | None,
        difficulty: str | None,
    ) -> pd.DataFrame:
        liked_indexes = [self.book_id_to_index[int(book_id)] for book_id in liked_book_ids if int(book_id) in self.book_id_to_index]
        query_scores = np.zeros(len(self.books), dtype=float)
        if query:
            query_scores = cosine_similarity(self.vectorizer.transform([query]), self.content_matrix).ravel()

        rows = []
        for book_index in range(len(self.books)):
            rows.append(
                {
                    **self._book_feature_dict(book_index),
                    "user_mean": 4.55 if liked_indexes else self.global_mean,
                    "user_count": float(max(1, len(liked_indexes))),
                    "genre_affinity": self._term_affinity(book_index, liked_indexes, "genres", set(genres or [])),
                    "mood_affinity": self._term_affinity(book_index, liked_indexes, "moods", set(moods or [])),
                    "author_affinity": self._author_affinity(book_index, liked_indexes),
                    "content_profile_similarity": self._average_similarity(book_index, liked_indexes, self.content_similarity),
                    "collaborative_profile_similarity": self._average_similarity(book_index, liked_indexes, self.collaborative_similarity),
                    "query_similarity": float(query_scores[book_index]),
                    "preference_match": self._single_preference_match(book_index, genres, moods, pace, length, difficulty),
                }
            )
        frame = pd.DataFrame(rows)
        return frame.reindex(columns=self.feature_columns, fill_value=0.0)

    def _book_feature_dict(self, book_index: int) -> dict[str, float]:
        row = self.books.iloc[book_index]
        stats = self.book_stats.loc[int(row["book_id"])]
        return {
            "year_norm": (float(row["year"]) - 1800.0) / 260.0,
            "pace_code": float(PACE_ORDER.get(str(row["pace"]), 1)) / 2.0,
            "length_code": float(LENGTH_ORDER.get(str(row["length"]), 1)) / 2.0,
            "difficulty_code": float(DIFFICULTY_ORDER.get(str(row["difficulty"]), 1)) / 2.0,
            "book_mean": float(stats["book_mean"]),
            "book_count_log": float(np.log1p(stats["book_count"])),
            "popularity_score": float(self.popularity_score[book_index]),
        }

    def _term_affinity(self, book_index: int, liked_indexes: list[int], column: str, extra_terms: set[str] | None = None) -> float:
        candidate_terms = set(str(self.books.iloc[book_index][column]).split(";"))
        profile_terms: set[str] = set(extra_terms or set())
        for liked_index in liked_indexes:
            profile_terms.update(str(self.books.iloc[liked_index][column]).split(";"))
        if not profile_terms:
            return 0.0
        return len(candidate_terms & profile_terms) / len(candidate_terms | profile_terms)

    def _author_affinity(self, book_index: int, liked_indexes: list[int]) -> float:
        if not liked_indexes:
            return 0.0
        author = str(self.books.iloc[book_index]["author"])
        liked_authors = {str(self.books.iloc[index]["author"]) for index in liked_indexes}
        return 1.0 if author in liked_authors else 0.0

    def _average_similarity(self, book_index: int, liked_indexes: list[int], matrix: np.ndarray) -> float:
        if not liked_indexes:
            return 0.0
        return float(matrix[book_index, liked_indexes].mean())

    def _single_preference_match(
        self,
        book_index: int,
        genres: Iterable[str] | None,
        moods: Iterable[str] | None,
        pace: str | None,
        length: str | None,
        difficulty: str | None,
    ) -> float:
        row = self.books.iloc[book_index]
        score = 0.0
        selected_genres = set(genres or [])
        selected_moods = set(moods or [])
        if selected_genres:
            score += len(selected_genres & set(str(row["genres"]).split(";"))) / len(selected_genres)
        if selected_moods:
            score += len(selected_moods & set(str(row["moods"]).split(";"))) / len(selected_moods)
        score += 0.5 if pace and pace != "Any" and row["pace"] == pace else 0.0
        score += 0.5 if length and length != "Any" and row["length"] == length else 0.0
        score += 0.5 if difficulty and difficulty != "Any" and row["difficulty"] == difficulty else 0.0
        return score / 3.5

    def _strategy_weights(self, strategy: str) -> tuple[float, float, float]:
        if strategy == "Familiar":
            return 0.72, 0.20, 0.08
        if strategy == "Exploratory":
            return 0.46, 0.34, 0.20
        return self.content_weight, self.collaborative_weight, self.popularity_weight

    def _preference_bonus(
        self,
        genres: Iterable[str] | None,
        moods: Iterable[str] | None,
        pace: str | None,
        length: str | None,
        difficulty: str | None,
    ) -> np.ndarray:
        genres = set(genres or [])
        moods = set(moods or [])
        bonus = np.zeros(len(self.books), dtype=float)
        for index, row in self.books.iterrows():
            book_genres = set(str(row["genres"]).split(";"))
            book_moods = set(str(row["moods"]).split(";"))
            if genres:
                bonus[index] += 0.28 * (len(genres & book_genres) / len(genres))
            if moods:
                bonus[index] += 0.22 * (len(moods & book_moods) / len(moods))
            if pace and pace != "Any":
                bonus[index] += 0.12 if row["pace"] == pace else -0.03
            if length and length != "Any":
                bonus[index] += 0.10 if row["length"] == length else -0.02
            if difficulty and difficulty != "Any":
                bonus[index] += 0.10 if row["difficulty"] == difficulty else -0.02
        return bonus

    def _format_result(
        self,
        index: int,
        score: float,
        liked_book_ids: list[int],
        genres: Iterable[str] | None,
        moods: Iterable[str] | None,
        pace: str | None,
        length: str | None,
        difficulty: str | None,
    ) -> RecommendationResult:
        row = self.books.iloc[index]
        reasons: list[str] = []
        if liked_book_ids:
            liked_titles = self.books[self.books["book_id"].isin(liked_book_ids)]["title"].head(2).tolist()
            reasons.append("similar to " + ", ".join(liked_titles))
        matched_genres = set(genres or []) & set(str(row["genres"]).split(";"))
        matched_moods = set(moods or []) & set(str(row["moods"]).split(";"))
        if matched_genres:
            reasons.append("matches " + ", ".join(sorted(matched_genres)))
        if matched_moods:
            reasons.append("feels " + ", ".join(sorted(matched_moods)))
        for label, value in [("pace", pace), ("length", length), ("difficulty", difficulty)]:
            if value and value != "Any" and row[label] == value:
                reasons.append(f"{value} {label}")
        if not reasons:
            reasons.append("strong overall reader signal")

        return RecommendationResult(
            book_id=int(row["book_id"]),
            score=float(score),
            title=str(row["title"]),
            author=str(row["author"]),
            year=int(row["year"]),
            genres=str(row["genres"]),
            moods=str(row["moods"]),
            pace=str(row["pace"]),
            length=str(row["length"]),
            difficulty=str(row["difficulty"]),
            description=str(row["description"]),
            explanation="; ".join(reasons),
        )
