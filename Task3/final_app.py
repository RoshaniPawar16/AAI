import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
import numpy as np

# Page config
st.set_page_config(
    page_title="MusicMind - Smart Music Recommendations",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton button {
        background-color: #76818e;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton button:hover {
        background-color: #5348d4;
    }
    .recommendation-card {
        background-color: #76818e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .recommendation-card:hover {
        transform: translateY(-5px);
    }
    .recommendation-card h3 {
        font-family: 'Poppins', sans-serif;
        color: #e7d7c1;
    }
    .recommendation-card p {
        font-family: 'Roboto', sans-serif;
        color: #262730;
    }
    .recommendation-card a {
        text-decoration: none;
        color: #e7d7c1;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
        transition: color 0.3s;
    }
    .youtube-link {
        background-color: #ff4b4b;
        color: white !important;
        padding: 8px 16px;
        border-radius: 20px;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Roboto', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .youtube-link:before {
        content: "▶";
        font-size: 0.8em;
    }
    .youtube-link:hover {
        background-color: #cc0000;
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(255, 0, 0, 0.2);
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("song_dataset.csv")
    return df

df = load_data()

@st.cache_resource
def run_imps(df):
    required_columns = ['user', 'song', 'play_count', 'title', 'artist_name', 'release']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")

    df = df.drop_duplicates(subset=['song', 'title', 'artist_name', 'release'])
    df['combined_features'] = (df['title'] + " " + df['artist_name'] + " " + df['release']).fillna("")

    # Content-Based Filtering
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    nn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='auto')
    nn.fit(tfidf_matrix)

    # Collaborative Filtering
    user_song_matrix = df.pivot_table(index='user', columns='song', values='play_count', fill_value=0)
    svd = TruncatedSVD(n_components=20)
    user_factors = svd.fit_transform(user_song_matrix)
    song_factors = svd.components_.T

    return df, tfidf, tfidf_matrix, nn, user_song_matrix, user_factors, song_factors

df, tfidf, tfidf_matrix, nn, user_song_matrix, user_factors, song_factors = run_imps(df)

# Content-based recommendation function
def content_based_recommend(song_title, top_n=5):
    try:
        idx = df[df['title'] == song_title].index[0]
        distances, indices = nn.kneighbors(tfidf_matrix[idx], n_neighbors=top_n + 1)
        song_indices = indices.flatten()[1:]
        return df.iloc[song_indices][['title', 'artist_name', 'release']].drop_duplicates()
    except IndexError:
        return pd.DataFrame(columns=['title', 'artist_name', 'release'])

def collaborative_recommend(user_id, top_n=5):
    if user_id not in user_song_matrix.index:
        return pd.DataFrame(columns=['title', 'artist_name', 'release'])

    user_vector = user_factors[user_song_matrix.index.get_loc(user_id)]
    scores = np.dot(song_factors, user_vector)

    listened_songs = user_song_matrix.loc[user_id][user_song_matrix.loc[user_id] > 0].index
    scores = {song: score for song, score in zip(user_song_matrix.columns, scores) if song not in listened_songs}

    recommended_songs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    recommended_song_ids = [song for song, _ in recommended_songs]
    return df[df['song'].isin(recommended_song_ids)][['title', 'artist_name', 'release']].drop_duplicates()

# Hybrid Recommendation
def hybrid_recommendv2(user_id, song_titles, top_n=5):
    collab_recs = collaborative_recommend(user_id, top_n)
    content_recs = pd.DataFrame()
    for song_title in song_titles:
        content_recs = pd.concat([content_recs, content_based_recommend(song_title, top_n)], ignore_index=True)
    hybrid_recs = pd.concat([collab_recs, content_recs]).drop_duplicates().sample(frac=1).reset_index(drop=True)
    return hybrid_recs.head(top_n)

# Sidebar and Main UI
with st.sidebar:
    st.header("🎯 Customize Your Recommendations")
    user_id = st.selectbox(
        "Select User ID",
        options=df['user'].unique(),
        index=0
    )
    user_songs = df[df['user'] == user_id]['title'].unique()
    song_title = st.multiselect(
        "Select Songs You Like",
        options=user_songs,
        default=user_songs[:1] if len(user_songs) > 0 else None
    )
    top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)
    get_recs = st.button("Get Recommendations! 🎶")

if get_recs:
    st.header("🎵 Your Recommendations")
    recommendations = hybrid_recommendv2(user_id, song_title, top_n)
    if recommendations.empty:
        st.error("No recommendations found. Try selecting different songs or users.")
    else:
        st.balloons()
        for idx, row in recommendations.iterrows():
            youtube_link = f"https://www.youtube.com/results?search_query={row['title']}+{row['artist_name']}"
            st.markdown(f"""
                <div class="recommendation-card">
                    <h3>{row['title']}</h3>
                    <p><strong>Artist:</strong> {row['artist_name']}</p>
                    <p><strong>Album:</strong> {row['release']}</p>
                    <a href="{youtube_link}" target="_blank" class="youtube-link">
                        Watch on YouTube
                </div>
            """, unsafe_allow_html=True)
