from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.data import ensure_data
from src.open_library import find_cover, search_books
from src.recommender import HybridBookRecommender, RecommendationResult


ARTIFACT_PATH = Path("artifacts/recommender.joblib")
FALLBACK_COVER = "https://covers.openlibrary.org/b/id/8231856-L.jpg"


st.set_page_config(
    page_title="ChapterWise",
    page_icon="CW",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource(show_spinner=False)
def load_model() -> HybridBookRecommender:
    ensure_data()
    if not ARTIFACT_PATH.exists():
        model = HybridBookRecommender()
        books, ratings = ensure_data()
        model.fit(books, ratings)
        model.save(ARTIFACT_PATH)
    return HybridBookRecommender.load(ARTIFACT_PATH)


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    books, _ = ensure_data()
    return books


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def cached_cover(title: str, author: str) -> str:
    return find_cover(title, author) or FALLBACK_COVER


@st.cache_data(show_spinner=False, ttl=60 * 15)
def cached_open_library_search(query: str, limit: int) -> list[dict[str, object]]:
    return search_books(query, limit)


def pills(values: str) -> str:
    return "".join(f"<span class='pill'>{escape(value.strip())}</span>" for value in values.split(";")[:4])


def score_label(score: float) -> str:
    normalized = max(0, min(100, int(round(score * 100))))
    return f"{normalized}% match"


def render_book_card(book: RecommendationResult) -> None:
    cover_url = cached_cover(book.title, book.author)
    st.markdown(
        f"""
        <article class="book-card">
            <div class="cover-wrap">
                <img src="{escape(cover_url)}" alt="{escape(book.title)} cover" />
            </div>
            <div class="book-copy">
                <div class="book-meta">
                    <span>{book.year}</span>
                    <span>{escape(book.pace)} pace</span>
                    <span>{escape(book.length)}</span>
                    <span>{escape(book.difficulty)} difficulty</span>
                </div>
                <h3>{escape(book.title)}</h3>
                <p class="author">by {escape(book.author)}</p>
                <p class="description">{escape(book.description)}</p>
                <div class="pill-row">{pills(book.genres)}</div>
                <div class="why">Why: {escape(book.explanation)}</div>
                <div class="match">{score_label(book.score)}</div>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_api_book(book: dict[str, object]) -> None:
    cover_url = str(book.get("cover_url") or FALLBACK_COVER)
    rating = book.get("rating")
    rating_text = f"{float(rating):.1f} avg" if rating else "Open Library"
    st.markdown(
        f"""
        <article class="api-card">
            <img src="{escape(cover_url)}" alt="{escape(str(book.get('title', 'Book')))} cover" />
            <div>
                <h3>{escape(str(book.get('title', 'Untitled')))}</h3>
                <p class="author">by {escape(str(book.get('author', 'Unknown author')))}</p>
                <p class="api-meta">{escape(str(book.get('year', '')))} · {escape(rating_text)}</p>
                <p class="api-subjects">{escape(str(book.get('subjects', '')))}</p>
                <a class="open-link" href="{escape(str(book.get('open_library_url', '#')))}" target="_blank">Open Library</a>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def option_map(books: pd.DataFrame) -> dict[str, int]:
    return {f"{row.title} - {row.author}": int(row.book_id) for row in books.itertuples()}


def unique_terms(books: pd.DataFrame, column: str) -> list[str]:
    return sorted({term.strip() for values in books[column] for term in str(values).split(";")})


def inject_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #f8f5ed;
                color: #1f2933;
            }
            .landing {
                min-height: 76vh;
                display: grid;
                grid-template-columns: minmax(0, 1.05fr) minmax(320px, 0.95fr);
                gap: 3rem;
                align-items: center;
                padding: 2.2rem 0 3rem;
                overflow: hidden;
            }
            .landing h1 {
                color: #18211f;
                font-size: clamp(3rem, 7vw, 6.8rem);
                line-height: 0.92;
                margin: 0;
                letter-spacing: 0;
            }
            .landing p {
                color: #465049;
                max-width: 720px;
                font-size: 1.08rem;
                line-height: 1.7;
                margin: 1.1rem 0 1.6rem;
            }
            .hero-actions {
                display: flex;
                flex-wrap: wrap;
                gap: 0.7rem;
            }
            .hero-chip {
                border: 1px solid #cfd9d5;
                background: rgba(255, 255, 255, 0.72);
                border-radius: 999px;
                padding: 0.52rem 0.82rem;
                font-size: 0.88rem;
                color: #24524d;
                font-weight: 700;
            }
            .cover-stage {
                position: relative;
                min-height: 520px;
            }
            .cover-stack {
                position: absolute;
                width: 190px;
                aspect-ratio: 2 / 3;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 22px 55px rgba(31, 41, 51, 0.2);
                animation: floaty 6s ease-in-out infinite;
                background: #d8c7a5;
            }
            .cover-stack img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }
            .cover-stack.one { left: 4%; top: 14%; transform: rotate(-8deg); }
            .cover-stack.two { left: 35%; top: 2%; transform: rotate(5deg); animation-delay: -1.5s; }
            .cover-stack.three { left: 54%; top: 34%; transform: rotate(10deg); animation-delay: -2.8s; }
            .cover-stack.four { left: 18%; top: 48%; transform: rotate(-3deg); animation-delay: -4s; }
            .orbit-line {
                position: absolute;
                inset: 5% 0 0 4%;
                border: 1px solid rgba(15, 118, 110, 0.22);
                border-radius: 50%;
                transform: rotate(-16deg);
                animation: slow-spin 18s linear infinite;
            }
            @keyframes floaty {
                0%, 100% { translate: 0 0; }
                50% { translate: 0 -18px; }
            }
            @keyframes slow-spin {
                from { rotate: 0deg; }
                to { rotate: 360deg; }
            }
            .section-head {
                padding: 1.5rem 0 1rem;
                border-bottom: 1px solid #ded6c9;
                margin-bottom: 1.25rem;
            }
            .section-head h2 {
                margin: 0;
                color: #18211f;
                font-size: 2.7rem;
                letter-spacing: 0;
            }
            .section-head p {
                color: #57534e;
                max-width: 820px;
            }
            .control-note {
                color: #57534e;
                font-size: 0.92rem;
                margin-bottom: 0.8rem;
            }
            .model-line {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin: 0.9rem 0 1.25rem;
            }
            .model-line span {
                display: inline-flex;
                align-items: center;
                min-height: 2rem;
                border: 1px solid #d8e3df;
                background: rgba(255, 255, 255, 0.68);
                border-radius: 999px;
                padding: 0.28rem 0.7rem;
                color: #134e4a;
                font-weight: 750;
                font-size: 0.82rem;
            }
            .page-divider {
                height: 1px;
                background: #ded6c9;
                margin: 1rem 0 2rem;
            }
            .book-card {
                min-height: 540px;
                border: 1px solid #ddd6c8;
                background: #fffefa;
                border-radius: 8px;
                overflow: hidden;
                position: relative;
                box-shadow: 0 10px 24px rgba(28, 25, 23, 0.06);
                transition: transform 180ms ease, box-shadow 180ms ease;
            }
            .book-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 18px 34px rgba(28, 25, 23, 0.11);
            }
            .cover-wrap {
                height: 250px;
                background: #e7dcc9;
            }
            .cover-wrap img {
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
            }
            .book-copy { padding: 1rem; }
            .book-card h3, .api-card h3 {
                font-size: 1.15rem;
                line-height: 1.25;
                margin: 0.5rem 0 0.15rem;
                color: #1c1917;
            }
            .author {
                color: #0f766e;
                margin: 0 0 0.65rem;
                font-weight: 700;
            }
            .description {
                color: #44403c;
                min-height: 6.3rem;
                font-size: 0.92rem;
                line-height: 1.45;
            }
            .book-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                color: #78716c;
                font-size: 0.72rem;
            }
            .book-meta span {
                border: 1px solid #ece3d6;
                border-radius: 999px;
                padding: 0.14rem 0.45rem;
                background: #fbf7ef;
            }
            .pill {
                display: inline-block;
                background: #e6f3f1;
                color: #115e59;
                border-radius: 999px;
                padding: 0.2rem 0.52rem;
                font-size: 0.75rem;
                font-weight: 700;
                margin: 0 0.25rem 0.25rem 0;
            }
            .why {
                border-top: 1px solid #ede6da;
                color: #57534e;
                margin-top: 0.8rem;
                padding-top: 0.7rem;
                padding-bottom: 2.4rem;
                font-size: 0.82rem;
            }
            .match {
                position: absolute;
                right: 0.9rem;
                bottom: 0.9rem;
                background: #134e4a;
                color: white;
                border-radius: 999px;
                padding: 0.3rem 0.65rem;
                font-size: 0.78rem;
                font-weight: 800;
            }
            .api-card {
                display: grid;
                grid-template-columns: 96px minmax(0, 1fr);
                gap: 0.9rem;
                border: 1px solid #ddd6c8;
                background: #fffefa;
                border-radius: 8px;
                padding: 0.75rem;
                min-height: 180px;
            }
            .api-card img {
                width: 96px;
                height: 146px;
                object-fit: cover;
                border-radius: 6px;
                background: #e6dcc9;
            }
            .api-meta, .api-subjects {
                color: #57534e;
                font-size: 0.86rem;
                margin: 0.35rem 0;
            }
            .open-link {
                color: #0f766e;
                font-weight: 800;
                text-decoration: none;
            }
            .live-strip {
                margin-top: 2rem;
                padding-top: 1.2rem;
                border-top: 1px solid #ded6c9;
            }
            .live-strip h3 {
                color: #18211f;
                font-size: 1.55rem;
                margin: 0 0 0.35rem;
            }
            .live-strip p {
                color: #57534e;
                margin: 0 0 1rem;
            }
            @media (max-width: 900px) {
                .landing { grid-template-columns: 1fr; }
                .cover-stage { min-height: 360px; }
                .cover-stack { width: 145px; }
                .control-layout { grid-template-columns: 1fr; }
            }
            @media (max-width: 640px) {
                .book-card { min-height: auto; }
                .cover-wrap { height: 220px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def landing_page(books: pd.DataFrame) -> None:
    hero_books = [
        ("Dune", "Frank Herbert"),
        ("The Hobbit", "J.R.R. Tolkien"),
        ("Pride and Prejudice", "Jane Austen"),
        ("Project Hail Mary", "Andy Weir"),
    ]
    covers = [cached_cover(title, author) for title, author in hero_books]
    st.markdown(
        f"""
        <section class="landing">
            <div>
                <h1>ChapterWise</h1>
                <p>Discover your next read through a trained recommendation engine that learns from reader ratings, nearest-neighbor similarity, book metadata, and live Open Library cover data.</p>
                <div class="hero-actions">
                    <span class="hero-chip">Supervised ML ranker</span>
                    <span class="hero-chip">KNN similarity</span>
                    <span class="hero-chip">Open Library covers</span>
                    <span class="hero-chip">{len(books)} starter books</span>
                </div>
            </div>
            <div class="cover-stage" aria-hidden="true">
                <div class="orbit-line"></div>
                <div class="cover-stack one"><img src="{escape(covers[0])}" alt="" /></div>
                <div class="cover-stack two"><img src="{escape(covers[1])}" alt="" /></div>
                <div class="cover-stack three"><img src="{escape(covers[2])}" alt="" /></div>
                <div class="cover-stack four"><img src="{escape(covers[3])}" alt="" /></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def recommendation_page(model: HybridBookRecommender, books: pd.DataFrame) -> None:
    options = option_map(books)
    all_genres = unique_terms(books, "genres")
    all_moods = unique_terms(books, "moods")

    selected_model = model.model_report.get("selected_model", "Supervised ranker")
    st.markdown(
        f"""
        <section class="section-head">
            <h2>Recommendation Engine</h2>
            <p>Pick books you already enjoy, tune the reading vibe, and get ranked recommendations with live Open Library cover data.</p>
        </section>
        <div class="model-line">
            <span>{len(books)} trained books</span>
            <span>{escape(str(selected_model))}</span>
            <span>{len(model.ratings):,} reader ratings</span>
            <span>KNN + TF-IDF retrieval</span>
            <span>Open Library linked</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_col, results_col = st.columns([0.34, 0.66], gap="large")
    with control_col:
        st.subheader("Controls")
        st.markdown("<p class='control-note'>Shape the reader profile and rerank the catalog instantly.</p>", unsafe_allow_html=True)
        liked_labels = st.multiselect(
            "Books you like",
            list(options.keys()),
            default=["Project Hail Mary - Andy Weir", "The Hobbit - J.R.R. Tolkien"],
            max_selections=6,
        )
        query = st.text_input("Search signal", placeholder="space politics, cozy fantasy, dark academia...")
        strategy = st.segmented_control("Strategy", ["Balanced", "Familiar", "Exploratory"], default="Balanced")
        selected_genres = st.multiselect("Genres", all_genres, default=[])
        selected_moods = st.multiselect("Moods", all_moods, default=[])
        pace = st.selectbox("Pace", ["Any", "slow", "medium", "fast"], index=0)
        length = st.selectbox("Length", ["Any", "short", "medium", "long"], index=0)
        difficulty = st.selectbox("Difficulty", ["Any", "low", "medium", "high"], index=0)
        top_n = st.slider("Results", 6, 18, 12, 3)

    liked_ids = [options[label] for label in liked_labels]
    recommendations = model.recommend(
        liked_book_ids=liked_ids,
        query=query,
        genres=selected_genres,
        moods=selected_moods,
        pace=pace,
        length=length,
        difficulty=difficulty,
        strategy=strategy,
        top_n=top_n,
    )

    with results_col:
        columns = st.columns(2)
        for index, book in enumerate(recommendations):
            with columns[index % 2]:
                render_book_card(book)

        live_query = query or " ".join(selected_genres[:2]) or "popular books"
        st.markdown(
            """
            <section class="live-strip">
                <h3>Live Open Library Matches</h3>
                <p>Fresh book metadata and covers from the public Open Library API, linked to your current search signal.</p>
            </section>
            """,
            unsafe_allow_html=True,
        )
        try:
            live_books = cached_open_library_search(live_query, 6)
        except requests.RequestException as exc:
            st.error(f"Open Library request failed: {exc}")
            live_books = []
        live_columns = st.columns(2)
        for index, book in enumerate(live_books):
            with live_columns[index % 2]:
                render_api_book(book)


def open_library_page() -> None:
    st.markdown(
        """
        <section class="section-head">
            <h2>Open Library Search</h2>
            <p>Fetch live books and cover art from the Open Library public API.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    api_query = st.text_input("Search Open Library", value="modern fantasy")
    limit = st.slider("Books to fetch", 6, 18, 12, 3)
    if api_query:
        try:
            books = cached_open_library_search(api_query, limit)
        except requests.RequestException as exc:
            st.error(f"Open Library request failed: {exc}")
            books = []
        columns = st.columns(2)
        for index, book in enumerate(books):
            with columns[index % 2]:
                render_api_book(book)


def main() -> None:
    inject_css()
    model = load_model()
    books = load_catalog()
    page = st.tabs(["Landing", "Recommendations", "Open Library"])
    with page[0]:
        landing_page(books)
    with page[1]:
        recommendation_page(model, books)
    with page[2]:
        open_library_page()


if __name__ == "__main__":
    main()
