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

# Recommend songs excluding those already listened to
def recommend_songs(listened_songs, user_factors, song_factors, song_id_to_title, n_recommendations=5):
    listened_song_ids = [title_to_song_id.get(title) for title in listened_songs if title in title_to_song_id]

    # Create a user-like average vector of the selected songs for better similarity
    if listened_song_ids:
        listened_vectors = [song_factors[list(user_item_matrix.columns).index(song)] for song in listened_song_ids]
        user_vector = np.mean(listened_vectors, axis=0)
    else:
        user_vector = np.mean(song_factors, axis=0)

    scores = np.dot(song_factors, user_vector)
    recommendations = [(song, score) for song, score in zip(user_item_matrix.columns, scores) if song not in listened_song_ids]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Map song IDs to titles and return
    return [song_id_to_title.get(song, song) for song, score in recommendations[:n_recommendations]]

# Streamlit UI
st.title("Custom Song Recommendation System")

# Load data and initialize model
data, song_id_to_title = load_data()
title_to_song_id = {title: song_id for song_id, title in song_id_to_title.items()}
user_item_matrix, user_factors, song_factors = create_recommendation_model(data)

# User input: Songs already listened to
st.sidebar.title("User Input")
song_titles = sorted(song_id_to_title.values())
listened_songs = st.sidebar.multiselect("Select songs you've already listened to", song_titles)

# Number of recommendations
n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Generate recommendations when the button is clicked
if st.sidebar.button("Recommend Songs"):
    recommendations = recommend_songs(listened_songs, user_factors, song_factors, song_id_to_title, n_recommendations)
    st.write("Recommended Songs:")
    for song in recommendations:
        st.write(f"- {song}")
