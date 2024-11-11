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

# Recommend similar songs to a given song
def recommend_similar_songs(song_title, song_factors, song_id_to_title, n_recommendations=5):
    song_id = title_to_song_id.get(song_title)
    if song_id is None:
        return ["Song not found"]

    # Find the song vector and compute similarities
    song_index = list(user_item_matrix.columns).index(song_id)
    song_vector = song_factors[song_index]
    similarities = cosine_similarity([song_vector], song_factors)[0]

    # Get top similar songs
    similar_songs = [(song_id_to_title.get(user_item_matrix.columns[i], user_item_matrix.columns[i]), similarities[i])
                     for i in np.argsort(similarities)[::-1] if user_item_matrix.columns[i] != song_id]
    
    # Return the top N similar songs
    return [song for song, score in similar_songs[:n_recommendations]]

# Streamlit UI
st.title("Song Recommendation System")

# Load data and initialize model
data, song_id_to_title = load_data()
title_to_song_id = {title: song_id for song_id, title in song_id_to_title.items()}
user_item_matrix, user_factors, song_factors = create_recommendation_model(data)

# User ID selection
st.sidebar.title("User Input")
user_id = st.sidebar.selectbox("Select User ID", data['user'].unique())

# Number of recommendations
n_recommendations = st.sidebar.slider("Number of Recommendations", 1, 10, 5)

# Step 1: Initial recommendations
if st.sidebar.button("Get Initial Recommendations"):
    recommendations = recommend_songs(user_id, user_item_matrix, user_factors, song_factors, song_id_to_title, n_recommendations)
    # Store recommendations in session state to persist them
    st.session_state['recommendations'] = recommendations

# Display recommendations if they exist in session state
if 'recommendations' in st.session_state:
    recommendations = st.session_state['recommendations']
    st.write(f"Recommended Songs for User {user_id}:")
    for song in recommendations:
        st.write(f"- {song}")

    # Allow user to select a song they liked from initial recommendations
    selected_song = st.selectbox("Select a song you liked from the recommendations", recommendations)

    # Step 2: Recommend similar songs to the selected song
    if selected_song:
        similar_songs = recommend_similar_songs(selected_song, song_factors, song_id_to_title, n_recommendations)
        st.write(f"Songs similar to '{selected_song}':")
        for song in similar_songs:
            st.write(f"- {song}")
