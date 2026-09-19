# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
data = {
    'movie_id' : [1,2,3,4,5],
    'title' : ['The Matrix','John Wick','The Godfather','Pulp fiction','The Dark Knight'],
    'genre' : ['Action,Sci-Fi','Action,Thriller','Crime,Drama','Crime,Drama','Action,Crime,Drama']
}
# Convert dataset into dataframe
df = pd.DataFrame(data)
# Display dataset
print("Movie Data")
# print(df)
# Define a Tf-IDF for vectorization of genre
tfidf = TfidfVectorizer(stop_words='english')
# Fit and transform the genre column into a matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(df['genre'])
# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix,tfidf_matrix)
# Function to recommend movies based on cosine similarity
def get_recommendations(title,cosine_sim=cosine_sim):
    # Get index of movie that matches the title
    idx = df[df['title'] == title].index[0]
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort movies based on similarity scores
    sim_scores = sorted(sim_scores,key = lambda x: x[1],reverse = True)
    # Get the indices of the 2 most similar movies
    sim_scores = sim_scores[1:3]
    # Get the movie indixes
    movie_indices = [i[0] for i in sim_scores]
    # Return titles of most similar movies
    return df['title'].iloc[movie_indices]
# Test the recommendation system with an example
movie_title = 'The Godfather'
recommended_movies = get_recommendations(movie_title)
print(f"Movie recommended for '{movie_title}: ")
for movie in recommended_movies:
    print(movie)