import streamlit as st
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('song_dataset.csv')  # Replace with the actual path to your file
    song_id_to_title = data.set_index('song')['title'].to_dict()  # Map song IDs to titles
    return data, song_id_to_title

# Create the user-item interaction matrix and SVD model
@st.cache_resource
def create_recommendation_model(data):
    user_item_matrix = data.pivot_table(index='user', columns='song', values='play_count', fill_value=0)
    svd = TruncatedSVD(n_components=50)
    user_factors = svd.fit_transform(user_item_matrix)
    song_factors = svd.components_.T
    return user_item_matrix, user_factors, song_factors

# Recommend songs for a given user
def recommend_songs(user_id, user_item_matrix, user_factors, song_factors, song_id_to_title, n_recommendations=5):
    if user_id not in user_item_matrix.index:
        return ["User not found"]

    listened_songs = user_item_matrix.loc[user_id]
    listened_songs = listened_songs[listened_songs > 0].index.tolist()
    user_vector = user_factors[user_item_matrix.index.get_loc(user_id)]
    scores = np.dot(song_factors, user_vector)

    recommendations = [(song, score) for song, score in zip(user_item_matrix.columns, scores) if song not in listened_songs]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Map song IDs to titles and return
    return [song_id_to_title.get(song, song) for song, score in recommendations[:n_recommendations]]

# Streamlit UI
st.title("Song Recommendation System")

# Load data and initialize model
data, song_id_to_title = load_data()
user_item_matrix, user_factors, song_factors = create_recommendation_model(data)

# User ID selection
st.sidebar.title("User Input")
user_id = st.sidebar.selectbox("Select User ID", data['user'].unique())

# Generate recommendations when the button is clicked
if st.sidebar.button("Recommend Songs"):
    recommendations = recommend_songs(user_id, user_item_matrix, user_factors, song_factors, song_id_to_title)
    st.write(f"Recommended Songs for User {user_id}:")
    for song in recommendations:
        st.write(f"- {song}")
