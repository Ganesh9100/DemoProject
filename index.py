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



pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
from transformers import pipeline

# Initialize the text generation pipeline with Gemma 3 1B
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to generate responses based on retrieved documents
def generate_response(query, retrieved_docs):
    # Combine the query with retrieved documents
    context = "\n".join([doc.text for doc in retrieved_docs])
    input_text = f"User: {query}\nContext: {context}\nAssistant:"

    # Generate the response
    response = text_generator(input_text, max_new_tokens=150)
    return response[0]['generated_text']


# Example user query
user_query = "What are the benefits of the premium plan?"

# Retrieve relevant documents
retriever = VectorIndexRetriever(index=index)
retrieved_documents = retriever.retrieve(user_query)

# Generate response using Gemma 3 1B
answer = generate_response(user_query, retrieved_documents)
print(answer)


