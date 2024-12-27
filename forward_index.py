import pandas as pd
import json
import os
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from collections import defaultdict

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')

# Pre-processing function to clean and tokenize text
def preprocess(text):
    # Handle possible NaN or null values in text
    if not isinstance(text, str):
        return []
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize text (split by whitespace)
    words = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return words

# Function to create the forward index from the text data
def create_forward_index(data):
    forward_index = defaultdict(list)
    
    for doc_id, text in enumerate(data['text']):  # Assuming 'text' is the column containing the content
        words = preprocess(text)
        for word in words:
            forward_index[word].append(doc_id)
    
    return forward_index

# Function to save the forward index to CSV and JSON
def save_forward_index(forward_index, output_dir):
    # Save to CSV
    forward_index_df = pd.DataFrame(list(forward_index.items()), columns=['Word', 'Document IDs'])
    csv_filename = os.path.join(output_dir, "forward_index.csv")
    forward_index_df.to_csv(csv_filename, index=False)
    print(f"Forward index saved to {csv_filename}")

    # Save to JSON
    json_filename = os.path.join(output_dir, "forward_index.json")
    with open(json_filename, 'w') as json_file:
        json.dump(forward_index, json_file)
    print(f"Forward index saved to {json_filename}")

# Main function to load data and generate the forward index
def main():
    # Load the dataset (update the path to your CSV file)
    file_path = "C:/Users/Abdul Aziz Zafar/OneDrive/Desktop/DSA Project/Project/medium_articles.csv"
    data = pd.read_csv(file_path, encoding='latin1', low_memory=False)  # Ensure 'text' column exists in your dataset
    
    # Create the forward index
    forward_index = create_forward_index(data)
    
    # Define output directory
    output_dir = "C:/Users/Abdul Aziz Zafar/OneDrive/Desktop/DSA Project/Project"
    
    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the forward index to files
    save_forward_index(forward_index, output_dir)

if __name__ == "__main__":
    main()
