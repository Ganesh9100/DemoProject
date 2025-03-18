import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
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

def chunk_updated(df):
    chunks = df.apply(lambda x: str(x.to_dict()),axis=1).to_list()
    return chunks
import pandas as pd
file_path = 'Data.csv'  # Update with your file path
df = load_data(file_path)
chunks = chunk_updated(df)
# Initialize the embedding model
print("Loading embedding model...")
s_model = SentenceTransformer('BAAI/bge-small-en-v1.5')


# Create embeddings for the text data
print("Creating embeddings...")
embeddings = model.encode(chunks, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')


# Create a FAISS index
print("Creating FAISS index...")
dimension = embeddings.shape[1]  # Get the dimension of embeddings
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)

# Add the vectors to the index
index.add(embeddings)
print(f"Added {index.ntotal} vectors of dimension {dimension} to the index")



def search_db(query_text,index, top_k=1):
    # Convert query to embedding
    query_embedding = s_model.encode([query_text], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype('float32')

    # Search the index
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": chunks[idx],
            "distance": float(distances[0][i]),
            "index": int(idx)
        })

    return results




sample_queries = [
#     "Extract monthly price under 40 dollar",
    "explain golden plan"
    
]
for query in sample_queries:
    print(f"\nQuery: '{query}'")
    results = search_db(query,index)
#     print("Top results:",results['text'])
    print()
    s = ''
    for i, result in enumerate(results):
#         print(f"{i+1}. [{result['distance']:.4f}] {result['text']}")
        s = s + result['text']
    print(s)
    
    
    
    
    
    
    
def generate_answer(model, tokenizer, context, question):
    """Generates an answer using the Gemma 3 1B model."""
    prompt = f"""
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    output = model.generate(**inputs, max_length=900,temperature=0.005,do_sample=True)
    return tokenizer.decode(output[0], skip_special_tokens=True)
def load_gemma_model():
    """Loads the Gemma 3 1B model and tokenizer."""
    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return model, tokenizer
def user_input(user_question,model, tokenizer):
    """Processes user input and retrieves an answer."""

    import time
    st = time.time()
    results = search_db(user_question,index)
    s_result = ''
    for i, result in enumerate(results):
#         print(f"{i+1}. [{result['distance']:.4f}] {result['text']}")
        s_result = s_result + result['text']

    print("similarity result",s_result)
    et = time.time()
    print("total time taken for fetcing from vector db",round(et-st,2))
    print()
    print()
    
    st = time.time()
    response = generate_answer(model, tokenizer, s_result, user_question)
    et = time.time()
    print("total time taken for llm",round(et-st,2))
    
    print("Gemma 3 1B Response:", response)
    
    
    
model, tokenizer = load_gemma_model()


while True:
    user_question = input("Ask a question (or type 'exit' to quit): ")
    if user_question.lower() == "exit":
        break
    user_input(user_question,model, tokenizer)
