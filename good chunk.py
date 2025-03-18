# index.py - Create FAISS index
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
with open('data.json', 'r') as f:
    data = json.load(f)

# Initialize model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Chunking function (customize if needed)
def chunk_text(text, max_length=512):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

# Prepare chunks with embeddings
texts, embeddings, metadata = [], [], []
for doc in data:
    for chunk in chunk_text(doc['text']):
        texts.append(chunk)
        embeddings.append(model.encode(chunk))
        metadata.append({'id': doc['id'], 'title': doc['title']})

embeddings = np.array(embeddings, dtype='float32')

# Create and save FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, 'faiss_index.bin')

# Save metadata
with open('metadata.json', 'w') as f:
    json.dump(metadata, f)

print("Indexing completed.")


# app.ipynb - Search using FAISS
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model, index, and metadata
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
index = faiss.read_index('faiss_index.bin')
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

def search(query, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype='float32'), top_k)

    results = []
    for i in range(len(indices[0])):
        results.append({
            'text': metadata[indices[0][i]]['title'],
            'distance': float(distances[0][i]),
        })
    return results

# Example usage
query = input("Enter your query: ")
print(search(query))
