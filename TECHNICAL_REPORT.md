# CINEIQ - Smart Movie Discovery

## Overview
CINEIQ is a next-generation movie recommendation engine designed to provide personalized, explainable, and sentiment-aware movie suggestions. By leveraging a hybrid approach that combines collaborative filtering (SVD), content-based filtering (TF-IDF), and advanced Natural Language Processing (VADER), CINEIQ delivers accurate recommendations while addressing the "black box" problem of modern algorithms.

## Architecture and Tech Stack
- **Frontend / Dashboard:** Streamlit (Python), Plotly for dynamic charts.
- **Backend API:** FastAPI for serving predictions and processing data.
- **Machine Learning Models:** Scikit-learn (TF-IDF, Cosine Similarity), Surprise (SVD for collaborative filtering).
- **Natural Language Processing:** NLTK (VADER Sentiment Analysis) on real audience reviews fetched from TMDB.
- **External APIs:** TMDB API for fetching movie metadata, posters, and audience reviews.

## Key Features
1. **Hybrid Recommendation Engine:** Combines the strengths of content similarity and user-item collaborative filtering for robust suggestions.
2. **Sentiment-Aware Re-Ranking:** Analyzes textual movie reviews using VADER to ensure recommended movies are genuinely well-received by audiences.
3. **Explainable AI (XAI) Layer:** Surfaces human-readable reasons for each recommendation, displaying match confidence and the signals used.
4. **Interactive Discovery:** Users can set minimum match confidence thresholds and save recommendations to a persistent watchlist.
5. **Taste DNA Dashboard:** Visualizes the user's genre affinity and temporal preferences using interactive radar and bar charts.

## Implementation Details
The application operates in two tiers:
1. **FastAPI Backend (`main.py`):** Exposes `/recommend` and `/user_stats` endpoints. It loads pre-computed similarity matrices and SVD models, computes hybrid scores for a given movie and user, and performs real-time sentiment analysis on fetched reviews.
2. **Streamlit Frontend (`app.py`):** Provides a modern, glassmorphic UI. It interacts with the FastAPI backend, renders movie posters, displays the XAI rationale, and renders user analytics via Plotly.

## Future Enhancements
- Integration of HuggingFace DistilBERT for more nuanced sentiment analysis.
- Transitioning from pickled models to a robust database (e.g., PostgreSQL or MongoDB) for scalability.
- User authentication and cloud-synced watchlists.
