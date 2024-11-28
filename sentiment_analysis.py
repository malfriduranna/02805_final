import os
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def load_sentiment_word_list(word_list_path):
    try:
        # Adjust the delimiter and skip metadata rows if necessary
        return pd.read_csv(word_list_path, sep="\t", comment='#', skip_blank_lines=True)
    except Exception as e:
        raise ValueError(f"Error reading word list: {e}")

def load_reviews(folder_path, sample_size=100):
    all_reviews = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)
            try:
                movie_reviews = pd.read_csv(file_path)
                sampled_reviews = movie_reviews.sample(n=sample_size, replace=False, random_state=42) \
                    if len(movie_reviews) > sample_size else movie_reviews
                sampled_reviews['movie'] = file_name.replace('.csv', '')
                all_reviews.append(sampled_reviews)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
    return pd.concat(all_reviews, ignore_index=True)

def compute_sentiment(review, happiness_dict):
    words = review.split()
    scores = [happiness_dict[word.lower()] for word in words if word.lower() in happiness_dict]
    return np.mean(scores) if scores else np.nan

def analyze_sentiments(folder_path, word_list_path, sample_size=100):
    sentiment_words = load_sentiment_word_list(word_list_path)
    happiness_dict = dict(zip(sentiment_words['word'], sentiment_words['happiness_average']))
    reviews_data = load_reviews(folder_path, sample_size)
    reviews_data['sentiment_score'] = reviews_data['review'].apply(lambda x: compute_sentiment(str(x), happiness_dict))
    reviews_data['sentiment_label'] = reviews_data['sentiment_score'].apply(
        lambda x: 'positive' if x >= 6 else ('negative' if x <= 4 else 'neutral')
    )
    sentiment_summary = reviews_data.groupby('movie').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        positive_count=('sentiment_label', lambda x: (x == 'positive').sum()),
        negative_count=('sentiment_label', lambda x: (x == 'negative').sum()),
        neutral_count=('sentiment_label', lambda x: (x == 'neutral').sum()),
        total_reviews=('review', 'count')
    )
    print(sentiment_summary)
    sentiment_summary['avg_sentiment'].plot(kind='bar', figsize=(12, 6), title='Average Sentiment by Movie')
    plt.xlabel('Movie')
    plt.ylabel('Average Sentiment')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return sentiment_summary

# Run the pipeline
folder_path = "2_reviews_per_movie_raw"  # Replace with the path to the folder containing the movie review files
word_list_path = "labMIT-1.0.txt"    # Replace with the path to the sentiment word list
sample_size = 100                    # Number of reviews to sample per movie

sentiment_results = analyze_sentiments(folder_path, word_list_path, sample_size)
