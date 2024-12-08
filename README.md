# AAI
Group Project CA1


TASK 3

Music-Mate : Smart Recommendations
Music-Mate is a music recommendation platform that combines content-based and collaborative filtering techniques to recommend songs to users based on their preferences. The app uses a dataset of songs and user play counts to suggest songs that a user might like.

Features
User-based recommendations: Suggests songs based on songs the user has already liked.
Song-based recommendations: Recommends similar songs based on a song's title, artist, and album.
YouTube links: Each recommended song has a link to its music video on YouTube.
Key Libraries and Tools Used
scikit-learn: Used for machine learning models like TF-IDF and K-Nearest Neighbors (KNN).
Requests: Used to fetch images from the PICTURE API for song artwork.
PICTURE API: Fetches unique song images based on the song's title and artist.
Streamlit: Used for the frontend to build a simple and interactive user interface.
pandas: Used for handling and manipulating the dataset.

Model Building and Data Preparation
The run_imps function prepares the dataset for the recommendation system:
It combines song features (title, artist, and release) into a single string to be used for content-based filtering.
It uses TF-IDF Vectorization to transform the text data (song features) into numerical data.
It creates a K-Nearest Neighbors (KNN) model to find the nearest songs based on their content features.
It also builds a user-song matrix for collaborative filtering, where each user’s song-play interactions are represented as a matrix, and another KNN model is used for collaborative filtering.

Content-Based Filtering
The content_based_recommend function provides song recommendations based on song titles selected by the user. It finds the closest songs to the selected song using the KNN model, based on the combined features of the song (title, artist, release year).

Collaborative Filtering
The collaborative_recommend function recommends songs to a user based on the songs that similar users have listened to. It uses the user-song matrix and the collaborative filtering model (KNN) to suggest songs that the user has not yet listened to but that similar users have enjoyed.

User Interface
The app displays a dropdown to select a user ID, a multiselect to choose songs the user likes, and a slider to select the number of recommendations.
Upon clicking the "Get Recommendations!" button, the app fetches song recommendations and displays them in a clean and organized manner.
For each recommended song, an image fetched from the Unsplash API is displayed, along with the song’s title, artist, album, and a link to watch the song on YouTube.
