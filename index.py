from llama_index.core import VectorStoreIndex
import json
from llama_index.core.schema import Document
from llama_index.core import SimpleDirectoryReader,ServiceContext,PromptTemplate
import faiss
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor




# Load data
with open("Data.json", "r") as f:
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


from llama_index.core.query_engine import RetrieverQueryEngine
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Initialize FAISS
dimension = 512  # Assuming OpenAI embedding dimension
index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=index)

# Initialize embedding model
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create Vector Store Index
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, vector_store=vector_store)

    
# Save FAISS Index
index.storage_context.persist("faiss_index")



