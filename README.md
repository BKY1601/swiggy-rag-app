# Swiggy Annual Report - RAG Question Answering System

## Objective
A Retrieval-Augmented Generation (RAG) based AI application designed to accurately answer user questions based ONLY on the [Swiggy Annual Report](https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf).The system strictly responds based on the document content to prevent hallucination.

## Dataset / Source Document
**Document Name:** swiggy_report
**Source Link:** [Swiggy Annual Report](https://www.swiggy.com/corporate/wp-content/uploads/2024/10/Annual-Report-FY-2023-24-1.pdf)

## Architecture & Hardware Optimization
This project was specifically engineered to run 100% locally on highly constrained hardware (Intel i5 6th Gen Dual-Core, 8GB RAM, No Dedicated GPU). To achieve this without API dependencies, the architecture utilizes highly optimized, quantized models and lightweight libraries.

**Document Processing:** Uses `PyMuPDF` for fast, memory-efficient PDF text extraction and LangChain's `RecursiveCharacterTextSplitter` to create manageable 500-character chunks.
**Embeddings & Vector Store:** Utilizes the lightweight `all-MiniLM-L6-v2` model via HuggingFace for generating embeddings, stored entirely in memory using `FAISS` for rapid semantic similarity search.
**Generator LLM:** Runs `qwen2.5:0.5b` via Ollama. This micro-model (0.5 billion parameters) operates completely locally with less than 1GB of RAM footprint while still strictly adhering to the retrieved context.

## Prerequisites
* Python 3.9+
* [Ollama](https://ollama.com/) installed and running locally.

## Setup & Installation

1. **Start the local LLM:**
   Open a terminal and run the Ollama model:
   ```bash
   ollama run qwen2.5:0.5b

(Keep this terminal open in the background)

2. **Set up the Python environment:**
  Open a new terminal and run:
   ```bash
   python -m venv venv
   venv\Scripts\activate

3. **Install dependencies:**
   ```bash
   pip install --no-cache-dir langchain langchain-classic langchain-community langchain-core pymupdf faiss-cpu sentence-transformers

4. **Add the Document:**
Ensure the Swiggy Annual Report is named swiggy_report.pdf and placed in the root directory.

## Usage

Run the application via the Command Line Interface (CLI):
  ```bash
  python app.py
```

The system will process the document, store the vectors, and open an interactive prompt. The interface outputs both the Final Answer and the Supporting Context retrieved from the PDF.

# Known Limitations & Future Scope
**Tabular Data parsing:** Due to the local hardware constraint of using a 0.5b parameter quantized model, the LLM occasionally struggles to align flattened rows and columns from complex financial tables, though the retriever successfully fetches the correct context chunks.

**Scalability:** For a production environment, this local RAG pipeline can be easily containerized and transitioned into a scalable Flask application backed by an API-based LLM.

## 👨‍💻 Author

**Bipin Yadav**  
📧 bipinyadav919@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/bipin-yadav-jan16)  
🔗 [GitHub](https://github.com/BKY1601)      
