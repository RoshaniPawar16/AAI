import pandas as pd
import streamlit as st

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('song_dataset.csv')

song_data = load_data()

# Recommendation function (placeholder)
def recommend_songs(user_input_songs):
    return song_data[~song_data['title'].isin(user_input_songs)].head(10)['title'].tolist()

# Streamlit app UI
st.title("Music Recommendation Engine")
user_songs = st.multiselect("Select songs you've listened to:", song_data['title'].unique())

if user_songs:
    recommendations = recommend_songs(user_songs)
    st.write("### Recommended Songs:")
    for song in recommendations:
        st.write(f"- {song}")
