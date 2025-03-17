import json
from llama_index import SimpleDirectoryReader, Document, VectorStoreIndex
from llama_index.vector_stores import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
import faiss

# Load data
with open("data.json", "r") as f:
    data = json.load(f)

# Function to create super chunks
def create_super_chunks(data):
    docs = []
    for duration, plans in data.items():
        combined_text = ""
        for plan in plans:
            # Extracting relevant details
            plan_name = plan.get("Plan Name", "")
            provider = plan.get("Provider Name", "")
            price = plan.get("Monthly Price", "").replace("$", "")  # Removing $
            network_speeds = f"5G Speed: {plan.get('5G-Typical Download Speed', '')}, {plan.get('5G-Typical Upload Speed', '')} | " \
                             f"4G Speed: {plan.get('4G LTE-Typical Download Speed', '')}, {plan.get('4G LTE-Typical Upload Speed', '')}"

            # Creating a combined text chunk
            combined_text += f"{provider} - {plan_name}\n{network_speeds}\n\n"

            # Store as a document with price metadata
            docs.append(Document(text=combined_text.strip(), metadata={"Monthly Price": float(price)}))
    
    return docs

# Generate super chunks
documents = create_super_chunks(data)


# Initialize FAISS
dimension = 1536  # Assuming OpenAI embedding dimension
index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=index)

# Initialize embedding model
embed_model = OpenAIEmbedding()

# Create Vector Store Index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, vector_store=vector_store)

# Save FAISS Index
index.storage_context.persist("faiss_index")


from llama_index.query_engine import RetrieverQueryEngine

# Load index
retriever = index.as_retriever(filters={"Monthly Price": {"<=": 50.0}})  # Example filter for plans below $50

# Query Engine
query_engine = RetrieverQueryEngine(retriever)

# Test Query
response = query_engine.query("Show me mobile plans with good 5G speeds under $50")
print(response)
