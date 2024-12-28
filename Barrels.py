import pandas as pd
import os
import nltk # type: ignore
from nltk.corpus import stopwords # type: ignore
from collections import defaultdict

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords')

# Pre-processing function to clean and tokenize text
def preprocess(text):
    if not isinstance(text, str):
        return []
    
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

# Function to create the inverted index (forward indexing)
def create_inverted_index(data):
    inverted_index = defaultdict(list)
    
    for doc_id, text in enumerate(data['text']):  # Assuming 'text' is the column containing the content
        words = preprocess(text)
        for word in set(words):  # Ensure each word is only added once per document
            inverted_index[word].append(doc_id)
    
    return inverted_index

# Function to create barrel indices (divide the index into barrels)
def create_barrel_indices(inverted_index, num_barrels=5):
    # Split the inverted index into barrels
    barrel_indices = defaultdict(list)
    
    # Divide words into barrels based on a hashing function
    for word, doc_ids in inverted_index.items():
        barrel_index = hash(word) % num_barrels  # Hash function to assign word to a barrel
        barrel_indices[barrel_index].append((word, doc_ids))
    
    return barrel_indices

# Function to save each barrel to CSV
def save_barrel_indices(barrel_indices, output_dir):
    for barrel_index, entries in barrel_indices.items():
        # Convert the entries into a DataFrame
        barrel_data = []
        for word, doc_ids in entries:
            barrel_data.append([word, ', '.join(map(str, doc_ids))])  # Join document IDs as a string
        
        barrel_df = pd.DataFrame(barrel_data, columns=['Word', 'Document IDs'])
        
        # Save the barrel to a CSV file
        barrel_filename = os.path.join(output_dir, f"barrel_{barrel_index}.csv")
        barrel_df.to_csv(barrel_filename, index=False)
        print(f"Barrel {barrel_index} saved to {barrel_filename}")

# Main function to load data and generate the barrel index
def main():
    # Load the dataset (update the path to your CSV file)
    file_path = "C:/Users/Abdul Aziz Zafar/OneDrive/Desktop/DSA Project/Project/medium_articles.csv"
    data = pd.read_csv(file_path, encoding='latin1', low_memory=False)  # Ensure 'text' column exists in your dataset
    
    # Create the inverted index
    inverted_index = create_inverted_index(data)
    
    # Create barrel indices
    num_barrels = 5  # You can change this based on how many barrels you want
    barrel_indices = create_barrel_indices(inverted_index, num_barrels)
    
    # Define output directory
    output_dir = "C:/Users/Abdul Aziz Zafar/OneDrive/Desktop/DSA Project/Project/Output"
    
    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the barrel indices to files
    save_barrel_indices(barrel_indices, output_dir)

if __name__ == "__main__":
    main()
