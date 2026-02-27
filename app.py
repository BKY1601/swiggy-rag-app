from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ---------------------------------------------------------
# 1. Document Processing
# ---------------------------------------------------------
print("Loading and chunking document...")
# PyMuPDF is very fast and easy on the CPU
loader = PyMuPDFLoader("swiggy_report.pdf")
docs = loader.load()

# Split text into chunks so we don't overwhelm the small model's memory
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

# ---------------------------------------------------------
# 2. Embedding & Vector Store
# ---------------------------------------------------------
print("Creating embeddings and vector store ...")
# Using a tiny, local, open-source embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Storing in FAISS (runs entirely in local RAM, very efficient)
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------------
# 3. RAG & LLM Setup
# ---------------------------------------------------------
print("Connecting to local Ollama model...")
# Connecting to the micro-model we downloaded
llm = Ollama(model="qwen2.5:0.5b")

# The prompt prevents hallucination 
prompt_template = """
You are an AI assistant. Answer the user's question using ONLY the provided context from the Swiggy Annual Report.
If the answer is not contained in the context, strictly reply: "I cannot answer this based on the provided document."

Context:
{context}

Question: {input}

Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# ---------------------------------------------------------
# 4. Question Answering Interface (CLI)
# ---------------------------------------------------------
print("\n--- Swiggy Annual Report RAG System Ready ---")
while True:
    user_query = input("\nAsk a question (or type 'quit' to exit): ")
    if user_query.lower() == 'quit':
        break
    
    print("Thinking...")
    # Get the answer from the chain
    response = rag_chain.invoke({"input": user_query})
    
    print("\n--- Final Answer ---")
    print(response["answer"])
    
    print("\n--- Supporting Context ---")
    for i, doc in enumerate(response["context"]):
        print(f"Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}): {doc.page_content[:150]}...")