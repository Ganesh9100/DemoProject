import json
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Load data
with open("Data.json", "r") as f:
    data = json.load(f)

# Function to create super chunks
def create_super_chunks(data):
    docs = []
    for duration, plans in data.items():
        combined_text = ""
        for plan in plans:
            plan_name = plan.get("Plan Name", "")
            provider = plan.get("Provider Name", "")
            price = plan.get("Monthly Price", "").replace("$", "")
            network_speeds = f"5G Speed: {plan.get('5G-Typical Download Speed', '')}, {plan.get('5G-Typical Upload Speed', '')} | " \
                             f"4G Speed: {plan.get('4G LTE-Typical Download Speed', '')}, {plan.get('4G LTE-Typical Upload Speed', '')}"
            combined_text += f"{provider} - {plan_name}\n{network_speeds}\n\n"
            docs.append(Document(page_content=combined_text.strip(), metadata={"Monthly Price": float(price)}))
    return docs

# Generate super chunks
documents = create_super_chunks(data)

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS index
dimension = 384  # Matching MiniLM embedding output size
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FAISS.from_documents(documents, embed_model, index=faiss_index)

# Save FAISS index
vector_store.save_local("faiss_index")
