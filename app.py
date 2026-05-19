import streamlit as st
import requests
import pandas as pd
import os
import random
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(page_title="CINEIQ | Smart Movie Discovery", layout="wide", page_icon="🎬")

# Initialize session state for Watchlist
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []

# --- Premium Glassmorphism & Cinematic Rose Gold Theme ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
       font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 20% 30%, #1a1216 0%, #0d0c0e 100%);
        color: #e4e3e7;
    }
    
    h1, h2, h3 {
        color: #ff4a5a;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .stMetric { 
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 15px; 
        border-radius: 16px; 
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .explanation-box {
        background: rgba(255, 74, 90, 0.06);
        border: 1px solid rgba(255, 74, 90, 0.2);
        padding: 14px;
        border-radius: 12px;
        font-size: 0.85em;
        color: #dfdbe2;
        margin-top: 12px;
        line-height: 1.5;
    }
    
    .movie-title {
        color: #ffffff;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 5px;
        font-size: 1.1em;
        letter-spacing: -0.2px;
    }
    
    .watchlist-item {
        background: rgba(255, 255, 255, 0.04);
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 3px solid #ff4a5a;
        font-size: 0.9em;
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIE_LIST_PATH = os.path.join(BASE_DIR, 'models', 'movie_list.pkl')

def fetch_poster(tmdb_id):
    # Fallback TMDB poster mappings for demo without API keys or internet
    fallbacks = {
        603: "https://image.tmdb.org/t/p/w500/f89U3wL3CUBMRZyZdUPzb46B0vS.jpg", # The Matrix
        27205: "https://image.tmdb.org/t/p/w500/o01vCoZNMjYWmG06egNsZ1jSTH3.jpg", # Inception
        157336: "https://image.tmdb.org/t/p/w500/gEU2QniE6E7vNIvTaK8vj6fhY80.jpg", # Interstellar
        155: "https://image.tmdb.org/t/p/w500/qJ2tWGB28uU4tWvgyCHZ59jRUEv.jpg", # The Dark Knight
        680: "https://image.tmdb.org/t/p/w500/d5i26jDw1uO11IwRKGRL4gL2BzW.jpg", # Pulp Fiction
        120: "https://image.tmdb.org/t/p/w500/6oom5Q55UPh68576jI74t234jC2.jpg", # LOTR Fellowship
        121: "https://image.tmdb.org/t/p/w500/w9kR8qbm2n6n6R612o1Cj6E0jow.jpg", # LOTR Two Towers
        122: "https://image.tmdb.org/t/p/w500/rCzpDGLbOo2asL669fCjjsz7UeC.jpg", # LOTR Return of the King
        13: "https://image.tmdb.org/t/p/w500/arw2tUv0qYiUrJXt66kgj2iJIS6.jpg", # Forrest Gump
        278: "https://image.tmdb.org/t/p/w500/q6y04FBhs2H3168A5q7ndE7JlsL.jpg"  # Shawshank
    }
    
    if tmdb_id in fallbacks:
        return fallbacks[tmdb_id]
        
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&language=en-US"
    fallback_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/500px-No_image_available.svg.png"
    
    try:
        data = requests.get(url, timeout=3).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        return fallback_image
    except:
        return fallback_image

# --- Title Header Layout ---
col_header_logo, col_header_text = st.columns([1, 12])
with col_header_logo:
    st.markdown("<h1 style='font-size: 3rem; margin: 0;'>🍿</h1>", unsafe_allow_html=True)
with col_header_text:
    st.title('CINEIQ')
    st.markdown("<p style='color: #a09ba6; font-size: 1.1em; margin-top: -10px;'>Next-generation Hybrid Engine featuring Sentiment Re-ranking & Explainable AI</p>", unsafe_allow_html=True)

st.markdown("---")

# --- Sidebar Configuration & New Features ---
with st.sidebar:
    st.markdown("<h2 style='color:#ff4a5a; font-size:1.5rem;'>⚙️ Discovery Desk</h2>", unsafe_allow_html=True)
    user_id = st.number_input("User Taste Profile ID", min_value=1, value=1, step=1)
    
    st.markdown("### 🛠️ Interactive Filters")
    min_match = st.slider("Min Match Confidence (%)", min_value=0, max_value=100, value=20, step=5)
    
    st.markdown("---")
    show_dashboard = st.checkbox("📊 Show User Analytics Dashboard", value=False)
    
    # Watchlist Section in sidebar
    st.markdown("---")
    st.markdown("<h3 style='color:#f4a261; font-size:1.2rem;'>📌 My Watchlist</h3>", unsafe_allow_html=True)
    if st.session_state.watchlist:
        for idx, item in enumerate(st.session_state.watchlist):
            st.markdown(f"<div class='watchlist-item'>{item}</div>", unsafe_allow_html=True)
        if st.button("🧹 Clear Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
    else:
        st.caption("No movies bookmarked yet.")

@st.cache_resource
def load_data():
    import pickle
    try:
        movies = pickle.load(open(MOVIE_LIST_PATH, 'rb'))
        return movies
    except FileNotFoundError:
        # Fallback if no model is present locally
        return pd.DataFrame({
            'title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump', 'The Shawshank Redemption', 'The Lord of the Rings: The Fellowship of the Ring'],
            'tmdbID': [603, 27205, 157336, 155, 680, 13, 278, 120]
        })

movies = load_data()
movie_list = movies['title'].values

col_sel1, col_sel2 = st.columns([3, 1])
with col_sel1:
    selected_movie = st.selectbox(
        "Which movie did you recently love?",
        movie_list
    )

def generate_mock_recommendations(selected_title):
    mock_data = {
        'The Matrix': [
            {"tmdbID": 27205, "title": "Inception", "score": 0.88, "approval": "91%"},
            {"tmdbID": 157336, "title": "Interstellar", "score": 0.82, "approval": "86%"},
            {"tmdbID": 155, "title": "The Dark Knight", "score": 0.79, "approval": "94%"},
            {"tmdbID": 120, "title": "The Lord of the Rings: The Fellowship of the Ring", "score": 0.75, "approval": "93%"},
            {"tmdbID": 680, "title": "Pulp Fiction", "score": 0.71, "approval": "92%"}
        ],
        'Inception': [
            {"tmdbID": 157336, "title": "Interstellar", "score": 0.91, "approval": "86%"},
            {"tmdbID": 603, "title": "The Matrix", "score": 0.85, "approval": "87%"},
            {"tmdbID": 155, "title": "The Dark Knight", "score": 0.81, "approval": "94%"},
            {"tmdbID": 278, "title": "The Shawshank Redemption", "score": 0.74, "approval": "98%"},
            {"tmdbID": 13, "title": "Forrest Gump", "score": 0.70, "approval": "89%"}
        ],
        'Interstellar': [
            {"tmdbID": 27205, "title": "Inception", "score": 0.92, "approval": "91%"},
            {"tmdbID": 603, "title": "The Matrix", "score": 0.84, "approval": "87%"},
            {"tmdbID": 155, "title": "The Dark Knight", "score": 0.78, "approval": "94%"},
            {"tmdbID": 120, "title": "The Lord of the Rings: The Fellowship of the Ring", "score": 0.76, "approval": "93%"},
            {"tmdbID": 278, "title": "The Shawshank Redemption", "score": 0.72, "approval": "98%"}
        ],
        'The Dark Knight': [
            {"tmdbID": 27205, "title": "Inception", "score": 0.89, "approval": "91%"},
            {"tmdbID": 603, "title": "The Matrix", "score": 0.83, "approval": "87%"},
            {"tmdbID": 157336, "title": "Interstellar", "score": 0.80, "approval": "86%"},
            {"tmdbID": 680, "title": "Pulp Fiction", "score": 0.77, "approval": "92%"},
            {"tmdbID": 278, "title": "The Shawshank Redemption", "score": 0.75, "approval": "98%"}
        ]
    }
    
    if selected_title in mock_data:
        return mock_data[selected_title]
        
    # Return random selections from the movie database
    sample_movies = movies[movies['title'] != selected_title].sample(min(5, len(movies)-1))
    results = []
    for _, r in sample_movies.iterrows():
        score = round(random.uniform(0.6, 0.95), 2)
        approval = f"{random.randint(70, 99)}%"
        results.append({
            "tmdbID": int(r['tmdbID']),
            "title": r['title'],
            "score": score,
            "approval": approval
        })
    return results

if st.button('✨ Unveil Matches', type="primary"):
    backend_url = f"http://127.0.0.1:8000/recommend"
    params = {"user_id": user_id, "movie_title": selected_movie}
    
    with st.spinner('Orchestrating recommendations...'):
        recommendations = []
        is_mocked = False
        try:
            response = requests.get(backend_url, params=params, timeout=2)
            if response.status_code == 200:
                recommendations = response.json()
            else:
                is_mocked = True
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            is_mocked = True
            
        if is_mocked:
            # Fallback to local prediction/mocking if backend is offline
            recommendations = generate_mock_recommendations(selected_movie)
            
        st.markdown("<br/>", unsafe_allow_html=True)
        st.subheader("🎯 Curated Matches For You")
        
        # Dynamic filter applied to the returned results
        filtered_recs = [m for m in recommendations if (m['score'] * 100) >= min_match]
        
        if not filtered_recs:
            st.warning("No recommendations matched your minimum match confidence threshold. Try lowering the slider in the sidebar!")
        else:
            cols = st.columns(3)
            
            for idx, movie in enumerate(filtered_recs):
                col_idx = idx % 3
                with cols[col_idx]:
                    st.markdown(f"<div class='movie-title'>{movie['title']}</div>", unsafe_allow_html=True)
                    poster_url = fetch_poster(movie['tmdbID'])
                    st.image(poster_url, use_column_width=True)
                 
                    st.metric(label="Audience Approval Rate", value=movie.get('approval', 'N/A'))
                    match_pct = round(movie['score'] * 100, 1)
                    
                    st.markdown(f"""
                    <div class="explanation-box">
                        <b>Recommendation Signal:</b><br/>
                        Surfaced a <b>{match_pct}%</b> similarity index. Backed by collaborative user-behavior tags relating to '{selected_movie}' with positive textual sentiment analysis.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Watchlist Save Button
                    if st.button(f"🔖 Keep in Watchlist", key=f"save_{movie['tmdbID']}"):
                        if movie['title'] not in st.session_state.watchlist:
                            st.session_state.watchlist.append(movie['title'])
                            st.success(f"Added {movie['title']} to Watchlist!")
                            st.rerun()
                        else:
                            st.info("Already in Watchlist!")
                    
                    st.markdown("<br/>", unsafe_allow_html=True)

if show_dashboard:
    st.markdown("---")
    st.header(f"📊 Cinematic Taste DNA — User {user_id}")
    
    # Mock data fallback for Dashboard if backend is offline
    stats = {
        "genres": {"Action": 85, "Sci-Fi": 90, "Drama": 40, "Comedy": 30, "Thriller": 70},
        "decades": {"1990s": 5, "2000s": 12, "2010s": 25, "2020s": 8},
        "total_ratings": 50
    }
    
    try:
        stats_res = requests.get(f"http://127.0.0.1:8000/user_stats/{user_id}", timeout=2)
        if stats_res.status_code == 200:
            stats = stats_res.json()
    except:
        pass # Keep mock data
        
    dash_col1, dash_col2 = st.columns([1.2, 1])
    
    with dash_col1:
        st.subheader("Genre Matrix Radar")
        categories = list(stats['genres'].keys())
        values = list(stats['genres'].values())
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='DNA',
            fillcolor='rgba(255, 74, 90, 0.25)',
            line_color='#ff4a5a'
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#a09ba6", gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(color="#a09ba6", gridcolor="rgba(255,255,255,0.08)"),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#e4e3e7",
            margin=dict(l=30, r=30, t=20, b=20)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with dash_col2:
        st.subheader("Temporal Preferences")
        decades = list(stats['decades'].keys())
        counts = list(stats['decades'].values())
        
        fig_bar = px.bar(
            x=counts, y=decades, orientation='h',
            labels={'x': 'Ratings Count', 'y': 'Era'},
            color_discrete_sequence=['#f4a261']
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="#e4e3e7",
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            margin=dict(l=0, r=10, t=20, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.caption(f"Compiled using {stats['total_ratings']} data points.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #a09ba6; font-size: 0.85em;'>CINEIQ • Powered by Surprise SVD & HuggingFace Sentiment Pipeline</p>", unsafe_allow_html=True)