# Movie Recommendation System

This project implements a movie recommendation system using two different approaches: **Surprise** and **PySpark**. The primary goal is to provide movie recommendations based on user ratings using collaborative filtering techniques.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Data](#data)
4. [Surprise Recommender System](#surprise-recommender-system)
5. [PySpark Recommender System](#pyspark-recommender-system)
6. [Results](#results)
7. [Usage](#usage)
8. [License](#license)

## Overview

This project explores two different recommendation algorithms for predicting user preferences for movies:

- **Surprise**: An easy-to-use Python library for building and evaluating recommender systems.
- **PySpark**: A scalable machine learning library in Apache Spark for handling large-scale data processing.

## Requirements

The following libraries are required to run this project:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `surprise`
- `pyspark`

You can install the necessary libraries using `pip`:

```bash
pip install numpy pandas scipy matplotlib seaborn surprise pyspark
```

## Data

The dataset used for this project is the [MovieLens dataset](https://grouplens.org/datasets/movielens/) which includes movie ratings and metadata. The following files are utilized:

- `ratings_small.csv`: Contains user ratings for movies.
- `movies_metadata.csv`: Contains metadata about movies.
- `links.csv`: Maps movies to their TMDB identifiers.

## Surprise Recommender System

The Surprise library is used for building a recommender system based on Singular Value Decomposition (SVD). The steps include:

1. **Loading Data**: The movie ratings data is loaded into a Pandas DataFrame.
2. **Data Visualization**: Distribution of ratings and average rating per user are plotted.
3. **Building Interaction Matrix**: An interaction matrix is created for the collaborative filtering model.
4. **Training SVD Model**: The SVD model is trained and evaluated using cross-validation.
5. **Saving and Loading Model**: The trained model is saved and loaded using the Surprise library.
6. **Generating Recommendations**: The top N recommendations for a specific user are generated and displayed.

## PySpark Recommender System

The PySpark library is used for large-scale data processing and building a recommendation model with Alternating Least Squares (ALS). The steps include:

1. **Loading Data**: The movie ratings data is loaded into a Spark DataFrame.
2. **Data Preparation**: User and movie IDs are mapped to unique integer values.
3. **Building ALS Model**: The ALS model is trained and evaluated using RMSE.
4. **Generating Recommendations**: Recommendations for a user are generated, and movie titles are fetched using metadata.

## Results

- **Surprise Recommender System**: Detailed evaluation metrics such as RMSE and MAE are provided. Top N recommendations for a user are displayed.
- **PySpark Recommender System**: RMSE for the ALS model is calculated, and top recommendations for a user are shown.

## Usage

1. **Surprise Recommender System**: Execute the script for the Surprise model to train the SVD algorithm and get recommendations.
2. **PySpark Recommender System**: Execute the script for the PySpark model to train the ALS algorithm and get recommendations.
