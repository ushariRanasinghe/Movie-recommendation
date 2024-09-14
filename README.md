# Movie Recommendation System

A movie recommendation system using collaborative filtering with Surprise and PySpark and a content-based kNN system. The collaborative filtering models recommend movies based on user preferences, leveraging past ratings to find similar users and suggest unseen movies. The kNN model uses keywords and genres to find movies similar to a user’s selected titles, offering quick suggestions based on content. The system includes a Streamlit frontend for movie search, selection, and recommendations.

Both approaches have distinct advantages: 

Collaborative Filtering excels at personalization, using user behavior patterns to make recommendations that may surprise the user but fit their tastes.

kNN shines in cold-start situations, where no historical user data is available. It bases recommendations on the content of the movies themselves, ensuring users get relevant results even with no prior behavior data.

## Features

- **Collaborative Filtering**:
  - **SVD** (Singular Value Decomposition) for collaborative filtering based on user ratings.
  - **ALS** (Alternating Least Squares) for collaborative filtering, leveraging user ratings for recommendations.

- **Content-based kNN**:
  - Utilizes **TF-IDF vectorization** on movie genres and keywords.
  - Implements **k-Nearest Neighbors (kNN)** to recommend similar movies based on user-selected movies.

- **Streamlit Frontend**:
  - Interactive user interface for movie search and selection.
  - Displays selected movie tags and provides recommendations based on user inputs.
  - Allows dynamic addition and removal of movie tags.
  - 
## Installation

1. Clone the repository
    
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run the Streamlit App

To start the Streamlit frontend, use the following command:

```bash
streamlit run app.py
```

###Streamlit interface
![image](https://github.com/user-attachments/assets/4d66a58f-ab44-4533-a2ee-1921ea86fa07)
![image](https://github.com/user-attachments/assets/48575d0c-9054-46e7-bf8c-afa2fa25d6a3)
![image](https://github.com/user-attachments/assets/d437b590-5c8f-4bd6-9b14-7ec9d9e15c50)


## Data

The dataset used for this project is the [MovieLens dataset](https://grouplens.org/datasets/movielens/) which includes movie ratings and metadata. The following files are utilized:

- `ratings_small.csv`: Contains user ratings for movies.
- `movies_metadata.csv`: Contains metadata about movies.
- `links.csv`: Maps movies to their TMDB identifiers.

