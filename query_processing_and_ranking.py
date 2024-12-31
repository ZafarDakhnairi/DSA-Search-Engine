import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# Load stop words once
stop_words = set(stopwords.words('english'))

# File path to your dataset
file_path = r'C:\Users\Abdul Aziz Zafar\OneDrive\Desktop\DSA Project\Project\medium_articles.csv'

# Function to preprocess text
def preprocess(text):
    if not isinstance(text, str):
        return ""  # Handle non-string values
    # Tokenize and remove stopwords
    words = [word for word in text.split() if word.lower() not in stop_words]
    return ' '.join(words)

# Function to process a query
def process_query(query, tfidf_vectorizer, tfidf_matrix):
    # Preprocess the query
    query = preprocess(query)
    query_vector = tfidf_vectorizer.transform([query])

    # Compute cosine similarity
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Rank documents based on similarity scores
    ranked_indices = scores.argsort()[::-1]  # Sort in descending order
    ranked_scores = scores[ranked_indices]

    return ranked_indices, ranked_scores

# Function to display results in chunks of 20
def display_results(query, tfidf_vectorizer, tfidf_matrix, data):
    ranked_indices, ranked_scores = process_query(query, tfidf_vectorizer, tfidf_matrix)

    total_results = len(ranked_indices)
    results_per_page = 20
    current_page = 0

    while current_page * results_per_page < total_results:
        # Calculate the start and end indices for the current page
        start_idx = current_page * results_per_page
        end_idx = min((current_page + 1) * results_per_page, total_results)

        print(f"\nShowing results {start_idx + 1} to {end_idx} of {total_results}:")
        
        # Display the results for the current page
        for rank, (index, score) in enumerate(zip(ranked_indices[start_idx:end_idx], ranked_scores[start_idx:end_idx]), start=1):
            if score > 0:  # Display only documents with a positive score
                title = data.iloc[index]['title']
                url = data.iloc[index]['url']
                print(f"{start_idx + rank}. {title} (Score: {score:.4f})")
                print(f"URL: {url}\n")
        
        # Ask the user if they want to see more results
        current_page += 1
        if current_page * results_per_page < total_results:
            user_input = input("Do you want to see more results? (y/n): ")
            if user_input.lower() != 'y':
                print("Exiting search.")
                break
        else:
            print("No more results.")
            break

# First Run: Preprocess and vectorize the text, then save to file
def first_run():
    print("Loading dataset...")
    try:
        # Load the dataset without specifying usecols first
        raw_data = pd.read_csv(file_path, encoding='latin1', low_memory=False)

        # Drop unnamed columns
        raw_data = raw_data.loc[:, ~raw_data.columns.str.contains('^Unnamed')]
        print("Available columns:", list(raw_data.columns))

        # Update column names based on what you see in the printed output
        titles_col = 'title'  # Replace with the actual column name for titles
        text_col = 'text'     # Replace with the actual column name for text
        url_col = 'url'       # Replace with the actual column name for URLs

        # Select only relevant columns
        data = raw_data[[titles_col, text_col, url_col]].copy()

    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

    print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Preprocess the text column
    print("Preprocessing text...")
    data['processed_text'] = data[text_col].fillna('').astype(str).str.split().apply(
        lambda words: ' '.join([word for word in words if word.lower() not in stop_words])
    )

    # Vectorize the processed text using TF-IDF
    print("Vectorizing text...")
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'])

    # Save the preprocessed data and vectorizer to a file
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump((tfidf_vectorizer, tfidf_matrix, data), f)
    
    print("Preprocessed data and vectorizer saved.")
    subsequent_runs()

# Subsequent Runs: Load the preprocessed data and vectorizer from file
def subsequent_runs():
    print("Loading preprocessed data...")
    with open('preprocessed_data.pkl', 'rb') as f:
        tfidf_vectorizer, tfidf_matrix, data = pickle.load(f)

    print("Preprocessed data and vectorizer loaded successfully.")

    # Main loop to handle multiple queries
    while True:
        query = input("Enter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("Exiting the search.")
            break
        else:
            display_results(query, tfidf_vectorizer, tfidf_matrix, data)

print("Starting the script...")

if not os.path.exists('preprocessed_data.pkl'):
    print("Preprocessed data not found. Running first run...")
    first_run()
else:
    print("Preprocessed data found. Running subsequent runs...")
    subsequent_runs()

print("Script execution completed.")
