import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_docs):
    """Extracts text from uploaded PDFs."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """Splits text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Creates and saves a FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_gemma_model():
    """Loads the Gemma 3 1B model and tokenizer."""
    model_name = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return model, tokenizer

def generate_answer(model, tokenizer, context, question):
    """Generates an answer using the Gemma 3 1B model."""
    prompt = f"""
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=500)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def user_input(user_question):
    """Processes user input and retrieves an answer."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    context = "\n".join([doc.page_content for doc in docs])
    model, tokenizer = load_gemma_model()
    response = generate_answer(model, tokenizer, context, user_question)
    print("Gemma 3 1B Response:", response)

def main():
    """Main function for processing documents and answering questions."""
    pdf_files = ["document1.pdf", "document2.pdf"]  # Replace with actual PDF paths
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    
    while True:
        user_question = input("Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break
        user_input(user_question)

if __name__ == "__main__":
    main()
