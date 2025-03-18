import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import pickle

# Load CSV data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Chunking based on payment frequency
def chunk_data(df, group_column='Payment Frequency'):
    grouped = df.groupby(group_column)
    chunks = []
    metadata = []
    for name, group in grouped:
        chunk_text = '\n'.join(group.apply(lambda x: str(x.to_dict()), axis=1))
        chunks.append(chunk_text)
        metadata.append({'Payment Frequency': name})
    return chunks, metadata

# Generate embeddings using sentence-transformers
def generate_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

# Create FAISS index and save locally using pickle
def create_faiss_index(embeddings, metadata, index_path='updated_faiss_index.pkl'):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    with open(index_path, 'wb') as f:
        pickle.dump((index, metadata), f)

# Perform search and return actual chunks
def search_faiss(query, index_path='updated_faiss_index.pkl', model_name='sentence-transformers/all-MiniLM-L6-v2', top_k=3):
    with open(index_path, 'rb') as f:
        index, metadata = pickle.load(f)

    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in range(len(indices[0])):
        result = {
            'chunk': metadata[indices[0][i]],
            'distance': distances[0][i]
        }
        results.append(result)
    return results

# if __name__ == '__main__':
file_path = 'Data.csv'  # Update with your file path
df = load_data(file_path)
chunks, metadata = chunk_data(df)
embeddings = generate_embeddings(chunks)
create_faiss_index(embeddings, metadata)

print("FAISS index created and saved using pickle!")

# Example search
query = "monthly payment plan"
results = search_faiss(query)
print("Search Results:", results)
