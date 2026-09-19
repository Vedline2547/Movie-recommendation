# 🎬 Movie Recommendation System

A simple **content-based movie recommendation system** built with Python and Scikit-learn.

The system recommends movies based on the similarity of their **genres**. It uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert movie genres into numerical vectors and **cosine similarity** to determine how similar the movies are.

## 📌 Project Overview

Movie recommendation systems are commonly used by platforms to help users discover movies they may enjoy.

In this project, I built a simple recommendation system that:

* Stores movie information in a Pandas DataFrame
* Processes movie genres using TF-IDF
* Calculates similarity between movies using cosine similarity
* Finds movies similar to a selected movie
* Returns the top 2 recommended movies

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Cosine Similarity

## 📂 Project Structure

```text
Movie-recommendation/
│
├── main.py
└── README.md
```

## 📊 Dataset

The project uses a small sample dataset containing:

* `movie_id` – Unique movie identifier
* `title` – Movie title
* `genre` – Movie genre information

Example movies include:

```text
The Matrix
John Wick
The Godfather
Pulp Fiction
The Dark Knight
```

## ⚙️ How It Works

### 1. Create the Dataset

The movie information is stored in a Pandas DataFrame.

```python
df = pd.DataFrame(data)
```

### 2. Convert Genres into TF-IDF Features

The `TfidfVectorizer` converts the movie genres into numerical feature vectors.

```python
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['genre'])
```

### 3. Calculate Cosine Similarity

Cosine similarity is used to measure how similar the movies are based on their genre features.

```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
```

A higher cosine similarity score means that two movies have more similar genre information.

### 4. Generate Recommendations

The recommendation function finds the selected movie, calculates its similarity with all other movies, sorts the results, and returns the two most similar movies.

```python
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:3]

    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]
```

## ▶️ Example

The system can be tested using:

```python
movie_title = 'The Matrix'

recommended_movies = get_recommendations(movie_title)

print(f"Movies recommended for '{movie_title}':")

for movie in recommended_movies:
    print(movie)
```

Example output:

```text
Movies recommended for 'The Matrix':
John Wick
The Dark Knight
```

The exact recommendations depend on the genre data and similarity scores.

## 🧠 Key Concepts Learned

Through this project, I practiced:

* Natural Language Processing (NLP)
* Text vectorization
* TF-IDF
* Cosine similarity
* Content-based recommendation systems
* Pandas DataFrames
* Scikit-learn
* Working with similarity matrices
* Building a simple recommendation algorithm

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Movie-recommendation.git
```

### 2. Navigate to the project

```bash
cd Movie-recommendation
```

### 3. Install the required libraries

```bash
pip install pandas scikit-learn
```

### 4. Run the program

```bash
python main.py
```

## 🔮 Future Improvements

This project can be expanded by:

* Adding a larger movie dataset
* Including movie descriptions and keywords
* Including actors and directors
* Adding user ratings
* Returning more recommendations
* Creating a web interface with Flask or Streamlit
* Building a hybrid recommendation system
* Using a real movie dataset such as TMDB

## 📚 What This Project Demonstrates

This project demonstrates how machine learning and NLP techniques can be used to build a basic recommendation system.

The overall workflow is:

```text
Movie Genres
     ↓
TF-IDF Vectorization
     ↓
Numerical Feature Matrix
     ↓
Cosine Similarity
     ↓
Similarity Scores
     ↓
Sort Movies
     ↓
Top Recommendations
```

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning & AI Enthusiast | Python Developer

---

⭐ If you find this project useful, feel free to star the repository!
