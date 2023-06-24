# Import necessary libraries
import os
import pandas as pd
from surprise import Reader, Dataset
from surprise import SVD
from surprise.model_selection import cross_validate

# Load the movie ratings dataset
ratings = pd.read_csv(os.path.join('data', 'movie_ratings.csv'))

# Define the Reader object to parse the dataset
reader = Reader(rating_scale=(1, 5))

# Load the dataset into Surprise's data format
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Split the data into training and testing sets
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()

# Build the collaborative filtering model (Singular Value Decomposition)
model = SVD()

# Train the model
model.fit(trainset)

# Evaluate the model
cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Make movie recommendations for a given user
user_id = 10
recommendations = []
movies_seen = set(ratings[ratings['user_id'] == user_id]['movie_id'])
for movie_id in trainset.all_items():
    if movie_id not in movies_seen:
        predicted_rating = model.predict(user_id, movie_id).est
        recommendations.append((movie_id, predicted_rating))
recommendations.sort(key=lambda x: x[1], reverse=True)
top_n = 10  # Number of top recommendations to display
for movie_id, predicted_rating in recommendations[:top_n]:
    test = ratings[ratings['movie_id'] == movie_id]
    movie_title = ratings[ratings['movie_id'] == movie_id]['movie_title'].iloc[0]
    print(f"Movie: {movie_title}, Predicted Rating: {predicted_rating}")

