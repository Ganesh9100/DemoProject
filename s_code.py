import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

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

# Create FAISS index
def create_faiss_index(embeddings,metadata, index_path='updated_faiss_index'):  

    
# Perform search and return actual chunks
def search_faiss(query, index, data, top_k=3):


# Load the index, embeddings, and data
def load_index(index_path="updated_faiss_index", data_path="plan_data.pkl"):


# if __name__ == '__main__':
file_path = 'Data.csv'  # Update with your file path
df = load_data(file_path)
chunks, metadata = chunk_data(df)
