# kNN Movie Recommendation System

A machine learning-based movie recommendation system using k-Nearest Neighbors (kNN) algorithm that analyzes user ratings to find similar movies. The system processes a dataset of 9,700+ movies to provide personalized movie recommendations.

## Overview

This project implements a collaborative filtering recommendation system using the k-Nearest Neighbors algorithm. It analyzes user ratings of movies to identify patterns and similarities between different films, allowing it to suggest movies that users might enjoy based on their previous preferences.

## Dataset

The system uses the MovieLens dataset containing:
- 9,742 movies with their titles and IDs
- 100,836 ratings from 610 users
- Each rating is on a scale of 0-5

## Implementation Details

### Data Preprocessing
- The system creates a user-movie matrix where each row represents a movie and each column represents a user
- Missing values (movies not rated by a user) are filled with zeros
- This sparse matrix is converted to a Compressed Sparse Row (CSR) matrix for efficient processing

### Model
- Uses scikit-learn's NearestNeighbors implementation
- Parameters:
  - Metric: Cosine similarity (measures the cosine of the angle between two vectors)
  - Algorithm: Brute force search
  - n_neighbors: 20 (finds the 20 most similar movies)

### Search Implementation
- Uses FuzzyWuzzy for fuzzy string matching to find movies even when titles are misspelled or incomplete
- The system:
  1. Takes a movie title input
  2. Finds the closest matching movie title in the database
  3. Locates this movie in the user-movie matrix
  4. Uses the kNN model to find similar movies based on user rating patterns
  5. Returns a list of recommended movies, excluding the input movie

## Usage

```python
# Sample usage
recommender('Jumanji', mat_movies, 10)
```

Output:
```
Movie Selected: Jumanji (1995) Index: 1
Searching for recommendations........
[[0.         0.41156227 0.45018189 0.45501892 0.46195443 0.47512358
  0.48183868 0.48438002 0.49254201 0.50243974]] [[  1 322 436 325 418 504 483 506 512  18]]
1                                         NaN
322                     Lion King, The (1994)
436                     Mrs. Doubtfire (1993)
325                          Mask, The (1994)
418                      Jurassic Park (1993)
504                         Home Alone (1990)
483    Nightmare Before Christmas, The (1993)
506                            Aladdin (1992)
512               Beauty and the Beast (1991)
18      Ace Ventura: When Nature Calls (1995)
```

## Requirements

- pandas
- scikit-learn
- scipy
- fuzzywuzzy (for fuzzy string matching)

## Future Improvements

- Add genre-based filtering to improve recommendations
- Implement evaluation metrics to assess recommendation quality
- Create a more user-friendly interface
- Handle the cold start problem for new users or movies
- Add content-based filtering to complement collaborative filtering
- Experiment with different similarity metrics and algorithms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sourced from the MovieLens dataset
- Inspired by collaborative filtering recommendation systems
