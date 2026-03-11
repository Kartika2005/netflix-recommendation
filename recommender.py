"""
Content-Based Recommendation Engine for Netflix titles.

Uses TF-IDF vectorisation on combined text features (genres, description,
director, cast, country, rating) and cosine similarity to surface titles
that are most similar to the user's watch history.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _combine_features(row: pd.Series) -> str:
    """Merge the text columns that describe a title into one string."""
    parts = [
        str(row.get("listed_in", "")),
        str(row.get("description", "")),
        str(row.get("director", "")),
        str(row.get("cast", "")),
        str(row.get("country", "")),
        str(row.get("rating", "")),
    ]
    return " ".join(p.strip() for p in parts if p.strip())


def build_feature_matrix(df: pd.DataFrame):
    """
    Build a TF-IDF feature matrix from the dataset.

    Returns
    -------
    tfidf_matrix : sparse matrix  (n_titles × n_features)
    vectorizer   : fitted TfidfVectorizer (kept in case we need it later)
    """
    combined = df.apply(_combine_features, axis=1).fillna("")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(combined)
    return tfidf_matrix, vectorizer


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def get_recommendations(
    df: pd.DataFrame,
    tfidf_matrix,
    watched_titles: list[str],
    n: int = 10,
) -> pd.DataFrame:
    """
    Given a list of watched/liked titles, return the top-*n* most similar
    titles that are **not** already in the watch list.

    The similarity is computed as the **average** cosine-similarity vector
    across all watched titles, so the result reflects the user's overall
    taste rather than a single title.

    Returns a DataFrame with columns:
        title, type, listed_in, rating, release_year, description, score
    sorted by descending score.
    """
    # Normalise title casing for robust matching
    title_lower = df["title"].str.lower()

    watched_lower = [t.lower() for t in watched_titles]
    watched_indices = [
        idx
        for idx, t in enumerate(title_lower)
        if t in watched_lower
    ]

    if not watched_indices:
        return pd.DataFrame(
            columns=["title", "type", "listed_in", "rating",
                      "release_year", "description", "score"]
        )

    # Average cosine similarity across all watched titles
    sim_scores = cosine_similarity(
        tfidf_matrix[watched_indices], tfidf_matrix
    ).mean(axis=0)

    # Build a helper series, then zero-out already-watched titles
    scores = pd.Series(sim_scores, index=df.index)
    scores.iloc[watched_indices] = -1  # exclude watched

    # Top-n indices
    top_indices = scores.nlargest(n).index

    result = df.loc[top_indices, ["title", "type", "listed_in",
                                  "rating", "release_year",
                                  "description"]].copy()
    result["score"] = scores.loc[top_indices].values
    result = result.sort_values("score", ascending=False)
    result = result.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Watch-history persistence (simple JSON file)
# ---------------------------------------------------------------------------

_DEFAULT_HISTORY_PATH = Path(__file__).parent / "watch_history.json"


def save_history(history: list[str], path: Path = _DEFAULT_HISTORY_PATH):
    """Save the watch-history list to a JSON file."""
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def load_history(path: Path = _DEFAULT_HISTORY_PATH) -> list[str]:
    """Load the watch-history list; returns [] if the file doesn't exist."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    return []
