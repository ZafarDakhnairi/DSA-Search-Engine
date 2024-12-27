import pandas as pd
import os
import json
from collections import Counter
from multiprocessing import Pool, cpu_count

# File path to the dataset
file_path = r"C:\Users\Abdul Aziz Zafar\OneDrive\Desktop\DSA Project\Project\medium_articles.csv"

# Directory to save the output files
output_dir = r"C:\Users\Abdul Aziz Zafar\OneDrive\Desktop\DSA Project\Project"

def tokenize_text(text):
    # Split text into words
    return text.split()

def process_chunk(chunk):
    # Tokenize the text column and count word frequencies
    counter = Counter()
    for text in chunk["text"]:
        if pd.notnull(text):  # Check for non-null values
            counter.update(tokenize_text(text))
    return counter

def build_lexicon_parallel(data, num_processes):
    # Split data into chunks for multiprocessing
    chunk_size = len(data) // num_processes
    chunks = [data[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    # Combine results from all processes
    combined_counter = Counter()
    for result in results:
        combined_counter.update(result)

    return combined_counter

if __name__ == "__main__":
    print("Reading dataset...")
    # Read the dataset
    data = pd.read_csv(file_path, encoding='latin1', usecols=["text"])

    print("Building lexicon...")
    # Build the lexicon using multiprocessing
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    lexicon_counter = build_lexicon_parallel(data, num_processes)

    # Convert the lexicon to a DataFrame for saving
    lexicon_df = pd.DataFrame(lexicon_counter.items(), columns=["word", "frequency"])

    # Save lexicon to CSV and JSON
    csv_path = os.path.join(output_dir, "lexicon.csv")
    json_path = os.path.join(output_dir, "lexicon.json")

    lexicon_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as json_file:
        json.dump(dict(lexicon_counter), json_file)

    print(f"Total unique words in lexicon: {len(lexicon_counter)}")
    print(f"Lexicon saved to {csv_path}")
    print(f"Lexicon saved to {json_path}")
