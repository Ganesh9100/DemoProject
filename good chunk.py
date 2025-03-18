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
def chunk_data(df, group_column='payment_frequency'):
    grouped = df.groupby(group_column)
    chunks = []
    metadata = []
    for name, group in grouped:
        chunk_text = '\n'.join(group.apply(lambda x: str(x.to_dict()), axis=1))
        chunks.append(chunk_text)
        metadata.append({'payment_frequency': name})
    return chunks, metadata

# Generate embeddings using sentence-transformers
def generate_embeddings(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks, show_progress_bar=True)
    return np.array(embeddings)

# Create FAISS index
def create_faiss_index(embeddings, metadata, index_path='faiss_index'):  
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    with open(f'{index_path}_metadata.json', 'w') as f:
        json.dump(metadata, f)
    print("FAISS index and metadata saved.")

# Search function
def search_faiss(query, model_name='sentence-transformers/all-MiniLM-L6-v2', index_path='faiss_index', top_k=5):
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    with open(f'{index_path}_metadata.json', 'r') as f:
        metadata = json.load(f)

    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    results = [{'metadata': metadata[i], 'distance': distances[0][j]} for j, i in enumerate(indices[0])]
    return results

if __name__ == '__main__':
    file_path = 'plans.csv'  # Update with your file path
    df = load_data(file_path)
    chunks, metadata = chunk_data(df)
    embeddings = generate_embeddings(chunks)
    create_faiss_index(embeddings, metadata)
    
    query = "Monthly payment plans"
    results = search_faiss(query)
    print("Search Results:", results)



pip install pandas faiss-cpu sentence-transformers transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def generate_response(query, top_k=5):
    # Retrieve relevant chunks using FAISS
    relevant_chunks = search_faiss(query, top_k=top_k)
    # Combine the retrieved chunks into a single prompt
    prompt = " ".join([chunk['metadata']['text'] for chunk in relevant_chunks])
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=50)
    # Decode and return the generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
query = "What are the details of the monthly payment plan?"
answer = generate_response(query)
print(answer)
