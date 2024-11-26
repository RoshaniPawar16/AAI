import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
import streamlit as st

# Music Recommender Class
class MusicRecommender:
    def __init__(self, df):
        self.df = df
        self.user_song_matrix = None
        self.song_similarity = None
        self.user_similarity = None
        self.content_similarity = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepare the data for recommendation"""
        # Check for missing columns and missing values
        required_columns = ['user', 'song', 'play_count', 'genre', 'tempo']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
            return

        if self.df[required_columns].isnull().sum().any():
            st.error("Missing values detected in the dataset. Please ensure all columns are filled.")
            return

        # Create user-song matrix
        self.user_song_matrix = self.df.pivot_table(
            index='user',
            columns='song',
            values='play_count',
            fill_value=0
        )

        # Calculate song similarity matrix (collaborative filtering)
        song_sparse = csr_matrix(self.user_song_matrix.T)
        self.song_similarity = cosine_similarity(song_sparse)

        # Calculate user similarity matrix
        user_sparse = csr_matrix(self.user_song_matrix)
        self.user_similarity = cosine_similarity(user_sparse)

        # Prepare content-based similarity matrix
        self._calculate_content_similarity()

    def _calculate_content_similarity(self):
        """Calculate song similarity based on genre and tempo"""
        # Extract relevant features for content-based filtering
        content_features = self.df[['genre', 'tempo']].drop_duplicates()

        # Handle missing tempo values
        content_features['tempo'] = content_features['tempo'].fillna(content_features['tempo'].mean())

        # One-hot encode genre and scale tempo
        content_features = pd.get_dummies(content_features, columns=['genre'])
        scaler = StandardScaler()
        content_features['tempo'] = scaler.fit_transform(content_features[['tempo']])

        # Calculate cosine similarity
        self.content_similarity = cosine_similarity(content_features)

        # Map similarity matrix to songs
        self.content_similarity = pd.DataFrame(
            self.content_similarity, 
            index=self.df['song'].drop_duplicates(),
            columns=self.df['song'].drop_duplicates()
        )

    def analyze_data(self):
        """Perform exploratory data analysis"""
        # Most listened songs
        song_popularity = self.df.groupby(['title', 'artist_name'])['play_count'].sum()\
                               .sort_values(ascending=False)

        # Most popular artists
        artist_popularity = self.df.groupby('artist_name')['play_count'].sum()\
                                   .sort_values(ascending=False)

        # User listening distribution
        user_distribution = self.df.groupby('user')['play_count'].sum()

        return song_popularity, artist_popularity, user_distribution

    def plot_insights(self):
        """Generate visualization plots"""
        song_popularity, artist_popularity, user_distribution = self.analyze_data()

        fig = plt.figure(figsize=(20, 15))

        # 1. Top 10 Songs
        plt.subplot(2, 2, 1)
        song_popularity.head(10).plot(kind='bar')
        plt.title('Top 10 Most Listened Songs')
        plt.xticks(rotation=45, ha='right')

        # 2. Top 10 Artists
        plt.subplot(2, 2, 2)
        artist_popularity.head(10).plot(kind='bar')
        plt.title('Top 10 Most Popular Artists')
        plt.xticks(rotation=45, ha='right')

        # 3. User Listen Count Distribution
        plt.subplot(2, 2, 3)
        sns.histplot(user_distribution, bins=50)
        plt.title('Distribution of User Listen Counts')
        plt.xlabel('Total Listen Count')
        plt.ylabel('Number of Users')

        # 4. Year-wise Song Distribution
        plt.subplot(2, 2, 4)
        sns.histplot(data=self.df, x='year', bins=30)
        plt.title('Distribution of Songs by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Songs')

        plt.tight_layout()
        return fig

    def get_song_recommendations(self, user_id, n_recommendations=5):
        """Get song recommendations for a user based on similarity of songs"""
        if user_id not in self.user_song_matrix.index:
            return "User not found in database"

        # Get user's listening history
        user_songs = self.user_song_matrix.loc[user_id]
        user_songs = user_songs[user_songs > 0].index

        # Find similar users
        user_idx = self.user_song_matrix.index.get_loc(user_id)
        similar_users = self.user_similarity[user_idx].argsort()[::-1][1:11]

        # Get songs from similar users
        recommendations = []
        for similar_user_idx in similar_users:
            similar_user_id = self.user_song_matrix.index[similar_user_idx]
            similar_user_songs = self.user_song_matrix.loc[similar_user_id]
            similar_user_songs = similar_user_songs[similar_user_songs > 0].index
            new_songs = [song for song in similar_user_songs if song not in user_songs]
            recommendations.extend(new_songs)

        # Get unique recommendations and sort by popularity
        recommendations = list(dict.fromkeys(recommendations))
        song_popularity = self.df.groupby('song')['play_count'].sum()
        recommendations = sorted(recommendations, 
                               key=lambda x: song_popularity[x], 
                               reverse=True)[:n_recommendations]

        # Get song details
        recommended_songs = self.df[self.df['song'].isin(recommendations)]\
            [['song', 'title', 'artist_name', 'year']].drop_duplicates()

        return recommended_songs

    def get_similar_songs(self, song_id, n_recommendations=5):
        """Get similar songs based on listening patterns"""
        if song_id not in self.user_song_matrix.columns:
            return "Song not found in database"

        song_idx = self.user_song_matrix.columns.get_loc(song_id)
        similar_scores = self.song_similarity[song_idx]
        similar_songs = self.user_song_matrix.columns[similar_scores.argsort()[::-1][1:n_recommendations+1]]

        similar_songs_df = self.df[self.df['song'].isin(similar_songs)]\
            [['song', 'title', 'artist_name', 'year']].drop_duplicates()

        return similar_songs_df

    def get_content_recommendations(self, song_id, n_recommendations=5):
        """Get song recommendations based on content features (genre, tempo)"""
        if song_id not in self.content_similarity.index:
            return f"Song {song_id} not found in content similarity matrix."

        # Get similar songs based on content similarity
        similar_scores = self.content_similarity.loc[song_id]
        similar_songs = similar_scores.sort_values(ascending=False).head(n_recommendations + 1).iloc[1:]

        # Fetch song details
        recommended_songs = self.df[self.df['song'].isin(similar_songs.index)]\
            [['song', 'title', 'artist_name', 'genre', 'tempo']].drop_duplicates()

        return recommended_songs

# Streamlit Web Application
def main():
    st.set_page_config(page_title="Music Recommendation System", layout="wide")
    
    st.title('Music Recommendation System')

    # Load data
    df = pd.read_csv('sample_SONG_Dataset.csv', encoding='ISO-8859-1')
    recommender = MusicRecommender(df)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Get Recommendations", "Dataset Stats", "Input Song Recommendations"])
    
    # Tab 1: Data Analysis
    with tab1:
        st.header("Data Analysis")
        song_popularity, artist_popularity, user_distribution = recommender.analyze_data()
        
        # Display top songs and artists
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 10 Most Listened Songs")
            st.dataframe(song_popularity.head(10))
        
        with col2:
            st.subheader("Top 10 Most Popular Artists")
            st.dataframe(artist_popularity.head(10))
        
        # Display plots
        st.pyplot(recommender.plot_insights())
    
    # Tab 2: Recommendations
    with tab2:
        st.header("Get Recommendations")
        
        # Create columns for different recommendation types
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("User-based Recommendations")
            user_id = st.selectbox(
                'Select User ID:',
                options=sorted(df['user'].unique())
            )
            if st.button('Get Recommendations for User'):
                recommendations = recommender.get_song_recommendations(user_id)
                if isinstance(recommendations, str):
                    st.write(recommendations)
                else:
                    st.dataframe(recommendations[['title', 'artist_name']])
        
        with col2:
            st.subheader("Song-based Recommendations")
            song_id = st.selectbox(
                'Select Song:',
                options=[(f"{row['title']} - {row['artist_name']}", row['song']) for _, row in df[['song', 'title', 'artist_name']].drop_duplicates().iterrows()]
            )
            if st.button('Get Similar Songs'):
                recommendations = recommender.get_similar_songs(song_id)
                st.dataframe(recommendations[['title', 'artist_name']])
    
    # Tab 3: Dataset Stats
    with tab3:
        st.header("Dataset Statistics")
        st.write(df.describe())
    
    # Tab 4: Content-based Recommendations
    with tab4:
        st.header("Content-based Recommendations")
        song_id = st.selectbox('Select Song for Content-based Recommendations', 
                               options=[(f"{row['title']} - {row['artist_name']}", row['song']) for _, row in df[['song', 'title', 'artist_name']].drop_duplicates().iterrows()])
        if st.button('Get Content-based Recommendations'):
            recommendations = recommender.get_content_recommendations(song_id)
            st.dataframe(recommendations[['title', 'artist_name']])

if __name__ == "__main__":
    main()
