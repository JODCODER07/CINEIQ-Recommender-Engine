# CINEIQ - Smart Movie Discovery

An open, explainable movie recommendation engine that combines collaborative filtering, content-based filtering, and sentiment-aware re-ranking to deliver personalized, interpretable suggestions that evolve with user taste.

## Problem Statement
Content discovery on modern streaming platforms is opaque, biased toward promoted titles, and can trap users in recommendation loops. **CINEIQ** aims to provide an open and explainable system that blends multiple ML strategies for better, more transparent recommendations.

## Deliverables
- **Hybrid Recommendation Engine**: Collaborative filtering + content-based filtering (TF‑IDF + cosine similarity) + SVD-based matrix factorization via weighted ensemble
- **Sentiment-Aware Re‑Ranker**: Uses VADER on user reviews to re-rank recommendations
- **User Taste Dashboard**: Streamlit interface with genre radar charts and decade preferences
- **Explainability Layer**: Human-readable recommendation rationale (rule-based XAI templates)

## Tech Stack
- **ML**: Python, scikit‑learn, Surprise (SVD), Pandas, NumPy
- **NLP**: NLTK VADER
- **Serving**: FastAPI
- **Dashboard**: Streamlit + Plotly
- **Tracking**: MLflow (installed; tracking optional)

## Repository Structure
```
.
├── app.py                 # Streamlit UI (Frontend)
├── main.py                # FastAPI backend (API Service)
├── filtering_content.py   # Data prep + model training script
├── TECHNICAL_REPORT.md    # Detailed project documentation
├── data/                  # Dataset folder (download required for local API)
├── models/                # Model artifacts folder (download required for local API)
├── .env.example
├── requirements.txt       # Frontend dependencies (lightweight, for Cloud deployment)
└── requirements-backend.txt # Backend and ML dependencies
```

## Setup
### 1) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2) Install dependencies
- To run the **Streamlit Dashboard (Frontend)**:
  ```bash
  pip install -r requirements.txt
  ```
- To run the **FastAPI Backend (API & Model Service)**:
  ```bash
  pip install -r requirements-backend.txt
  ```

### 3) Configure environment
Create a `.env` file using `.env.example`:
```bash
TMDB_API_KEY=YOUR_TMDB_KEY
```

---

## ⚡ Mock & Offline Fallback (For Quick Testing)
If the FastAPI backend is not running or the model pickle files are not downloaded, the **Streamlit Dashboard** will automatically fall back to an interactive **Mock Mode**. 
- Allows you to test the UI, filters, watchlist, and sample charts immediately without downloading gigabytes of dataset files.

---

## Running the Project Locally
### 1) Start the FastAPI backend
If you have downloaded the datasets and models (see below):
```bash
uvicorn main:app --reload
```
The API will run at `http://127.0.0.1:8000`.

### 2) Start the Streamlit dashboard
```bash
streamlit run app.py
```
The UI will open at `http://localhost:8501`.

---

## Downloading Datasets & Models (For full local API execution)
Large files are hosted on Google Drive.

### ✅ Datasets
Download from:
- https://drive.google.com/drive/folders/15-w7RubqiwIjvltmjiMaPQ44gQYJ1HAH?usp=sharing

Place the following in `data/`:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`
- `links.csv`
- `ratings.csv`

### ✅ Models
Download from:
- https://drive.google.com/drive/folders/1BsvU_4uSxTSvKv2ysKovJufq0Lyeqqq2?usp=sharing

Place the following in `models/`:
- `movie_list.pkl`
- `similarity.pkl`
- `svd_model.pkl`
- `user_ratings.pkl`

---

## API Endpoints
- `GET /recommend?user_id=...&movie_title=...`
- `GET /user_stats/{user_id}`
