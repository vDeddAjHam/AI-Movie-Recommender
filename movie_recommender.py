from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split, accuracy
import pandas as pd

def load_movie_data():
    # For demonstration, using a small sample dataset
    data_dict = {
        "user": ["A", "A", "B", "B", "C", "C"],
        "item": ["Movie1", "Movie2", "Movie1", "Movie3", "Movie2", "Movie3"],
        "rating": [4, 5, 3, 4, 2, 5]
    }
    df = pd.DataFrame(data_dict)
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['user', 'item', 'rating']], reader)

def train_recommender(data):
    trainset = data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

def get_movie_recommendation(algo, user_id, items, n=3):
    predictions = []
    for item in items:
        pred = algo.predict(user_id, item)
        predictions.append((item, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]

if __name__ == "__main__":
    data = load_movie_data()
    algo = train_recommender(data)
    user = input("Enter user ID (e.g., A): ")
    all_movies = ["Movie1", "Movie2", "Movie3"]
    recommendations = get_movie_recommendation(algo, user, all_movies)
    print("Top movie recommendations for user", user)
    for movie, rating in recommendations:
        print(f"{movie} (predicted rating: {rating:.2f})")
