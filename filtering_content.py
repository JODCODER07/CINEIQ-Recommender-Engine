import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
import os
import pickle

# Set base paths for datasets
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_METADATA_CSV = os.path.join(BASE_DIR, 'data', 'tmdb_5000_movies.csv')
CREDITS_CSV = os.path.join(BASE_DIR, 'data', 'tmdb_5000_credits.csv')
LINKS_CSV = os.path.join(BASE_DIR, 'data', 'links.csv')
RATINGS_CSV = os.path.join(BASE_DIR, 'data', 'ratings.csv')

def parse_metadata_list(raw_json):
    """Converts a raw JSON string into a list of name attributes."""
    items = []
    try:
        for item in ast.literal_eval(raw_json):
            items.append(item['name'])
    except Exception:
        pass
    return items

def extract_movie_director(raw_crew_json):
    """Parses crew JSON to find and extract the Director's name."""
    directors = []
    try:
        for crew_member in ast.literal_eval(raw_crew_json):
            if crew_member.get('job') == 'Director':
                directors.append(crew_member['name'])
    except Exception:
        pass
    return directors

def strip_spaces(names_list):
    """Strips whitespaces inside strings to create singular tokens for TF-IDF tag extraction."""
    return [name.replace(" ", "") for name in names_list]

def train_recommender_models():
    """Reads raw datasets, prepares structural textual features, trains SVD/TF-IDF models and saves pickle files."""
    print("Loading raw CSV files from data directory...")
    if not (os.path.exists(MOVIES_METADATA_CSV) and os.path.exists(CREDITS_CSV) and os.path.exists(LINKS_CSV) and os.path.exists(RATINGS_CSV)):
        print("Error: Missing dataset CSV files in the 'data/' folder. Please download the datasets first.")
        return
        
    movies_df = pd.read_csv(MOVIES_METADATA_CSV)
    credits_df = pd.read_csv(CREDITS_CSV)
    links_df = pd.read_csv(LINKS_CSV)
    
    ratings_dtype = {'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'}
    ratings_df = pd.read_csv(RATINGS_CSV, usecols=['userId', 'movieId', 'rating'], dtype=ratings_dtype)
    # Take a representative 20% sample for performance speed and memory efficiency
    ratings_sampled = ratings_df.sample(frac=0.2, random_state=42)

    # Data Processing & Cleansing
    links_df.dropna(inplace=True)
    links_df["tmdbId"] = links_df["tmdbId"].astype(int)
    
    # Merge movies with credits
    merged_movies = movies_df.merge(credits_df, on="title")
    merged_movies = merged_movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    
    # Merge with links to resolve movieLens and TMDB mappings
    aligned_movies = merged_movies.merge(links_df[['movieId', 'tmdbId']], left_on='movie_id', right_on='tmdbId', how='inner')
    aligned_movies.drop(columns=['tmdbId'], inplace=True)
    aligned_movies.rename(columns={"movie_id": "tmdbID"}, inplace=True)
    aligned_movies.dropna(inplace=True)

    # Parse JSON attributes
    aligned_movies['genres'] = aligned_movies['genres'].apply(parse_metadata_list)
    aligned_movies['keywords'] = aligned_movies['keywords'].apply(parse_metadata_list)
    aligned_movies['cast'] = aligned_movies['cast'].apply(parse_metadata_list).apply(lambda x: x[0:10])
    aligned_movies['crew'] = aligned_movies['crew'].apply(extract_movie_director)

    # Strip whitespaces to prepare for content tag vectorization
    aligned_movies['cast'] = aligned_movies['cast'].apply(strip_spaces)
    aligned_movies['crew'] = aligned_movies['crew'].apply(strip_spaces)
    aligned_movies['genres'] = aligned_movies['genres'].apply(strip_spaces)
    aligned_movies['keywords'] = aligned_movies['keywords'].apply(strip_spaces)
    
    # Split text descriptions into list of words
    aligned_movies['overview'] = aligned_movies['overview'].apply(lambda x: x.split())
    
    # Concatenate features into single tags column
    aligned_movies['tags'] = (aligned_movies['overview'] + 
                              aligned_movies['genres'] + 
                              aligned_movies['keywords'] + 
                              aligned_movies['cast'] + 
                              aligned_movies['crew'])
    
    processed_movies = aligned_movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
    processed_movies['tags'] = processed_movies['tags'].apply(lambda x: " ".join(x))

    # Perform TF-IDF Vectorization & Cosine Similarity for Content-based filtering
    print("Computing TF-IDF matrices & Cosine similarity...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrices = tfidf_vectorizer.fit_transform(processed_movies['tags'])
    cosine_sim_matrix = cosine_similarity(tfidf_matrices)

    # Train Collaborative Filtering SVD Model
    print("Fitting SVD matrix factorization collaborative engine...")
    rating_reader = Reader(rating_scale=(0.5, 5.0))
    ratings_dataset = Dataset.load_from_df(ratings_sampled[['userId', 'movieId', 'rating']], rating_reader)
    full_trainset = ratings_dataset.build_full_trainset()
    
    collaborative_svd = SVD(n_factors=50, lr_all=0.005, reg_all=0.02, random_state=42)
    collaborative_svd.fit(full_trainset)

    # Serialize and Save Models
    models_dir = os.path.join(BASE_DIR, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")

    pickle.dump(processed_movies, open(os.path.join(models_dir, 'movie_list.pkl'), 'wb'))
    pickle.dump(cosine_sim_matrix, open(os.path.join(models_dir, 'similarity.pkl'), 'wb'))
    pickle.dump(collaborative_svd, open(os.path.join(models_dir, 'svd_model.pkl'), 'wb'))
    pickle.dump(ratings_sampled, open(os.path.join(models_dir, 'user_ratings.pkl'), 'wb'))
    
    print("Successfully generated and saved all model pickle files to /models directory.")

if __name__ == "__main__":
    train_recommender_models()
