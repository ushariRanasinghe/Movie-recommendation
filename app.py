import numpy as np
import pandas as pd
import streamlit as st
import joblib
from streamlit_searchbox import st_searchbox
from typing import List

# Load the pre-trained kNN model, TF-IDF matrix and csv files
knn = joblib.load('streamlit/knn_model.pkl')
X = joblib.load('streamlit/tfidf_matrix.pkl')
Gdf = pd.read_csv('streamlit/Gdf.csv')
movie_df = pd.read_csv('streamlit/movie_df.csv')

#initialize session state
if 'selected_movie_ids' not in st.session_state:
    st.session_state.selected_movie_ids = []
if 'selected_movie_titles' not in st.session_state:
    st.session_state.selected_movie_titles = []
if 'flag' not in st.session_state:
    st.session_state.flag=''

#function to recommend movies using kNN model
def recommend_movies(input_movie_ids, n_recommendations=10):
    movie_indices = [
        movie_df.index.get_loc(movie_df[movie_df['id'] == movie_id].index[0])
        for movie_id in input_movie_ids
    ]
    input_vectors = X[movie_indices].toarray()
    combined_vector = np.mean(input_vectors, axis=0)
    distances, indices = knn.kneighbors([combined_vector],
                                        n_neighbors=n_recommendations +
                                        len(input_movie_ids))
    similar_movies = movie_df.iloc[indices[0]]
    similar_movies = similar_movies[~similar_movies['id'].isin(input_movie_ids
                                                               )]
    similar_movies = similar_movies.merge(Gdf[['id', 'title', 'overview']],
                                          on='id',
                                          how='left')
    return similar_movies[['id', 'title', 'overview']]


# Function to recommend search terms
def search_movies(searchterm: str) -> List[str]:
    if not searchterm:
        return []
    searchterm_lower = searchterm.lower()
    results = Gdf[Gdf['title'].str.contains(searchterm_lower,
                                            case=False,
                                            na=False)]
    top_results = results.head(10)['title'].tolist()
    return top_results


# Streamlit app layout
st.title("Movie Recommendation System")

# Pass the custom search function to st_searchbox
selected_value = st_searchbox(
    search_movies,
    placeholder="Search for movies...",
    clear_on_submit=True,
)

#Add selected movie title to movie titile list
if selected_value and selected_value!=st.session_state.flag:
    st.session_state.flag=selected_value
    matching_movie = Gdf[Gdf['title'].str.contains(selected_value,
                                                   case=False,
                                                   na=False)]
    if not matching_movie.empty:
        selected_movie_id = matching_movie['id'].values[0]
        if selected_movie_id not in st.session_state.selected_movie_ids:
            st.session_state.selected_movie_ids.append(selected_movie_id)
            st.session_state.selected_movie_titles.append(
                matching_movie['title'].values[0])

# Clear selected movies and refresh the page
if st.button("Clear All Selected Movies"):
    st.session_state.selected_movie_ids = []
    st.session_state.selected_movie_titles = []
    st.rerun() 

# Display the currently selected movies as tags
if st.session_state.selected_movie_titles:
    st.subheader("Selected Movies:")
for idx, title in enumerate(st.session_state.selected_movie_titles):
    if st.button(f"Remove {title}", key=f"remove_{title}"):

        st.session_state.selected_movie_titles.pop(idx)
        st.session_state.selected_movie_ids.pop(idx)

        st.rerun() 

#Display movie recommendations
if st.session_state.selected_movie_ids:
    recommended_movies = recommend_movies(st.session_state.selected_movie_ids)
    st.subheader("Recommended Movies:")
    st.write(recommended_movies[['title', 'overview']])

