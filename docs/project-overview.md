## **Project Plan: RAG MVP Demo**

### **Project Goal**

We'll build a simple **Python application** that demonstrates the core concept of Retrieval-Augmented Generation (RAG). The final product will be a web service, built with **FastAPI**, that can read a PDF document and accurately answer questions about its content, citing the exact line numbers for its sources.

The main goal remains to create a clear, easy-to-understand demonstration with two primary functions:

1.  **Ingesting a document.**
2.  **Answering questions based on that document.**

---

### **Core Technologies**

- **Language:** Python
- **Web Framework:** FastAPI
- **PDF Reading:** `pypdf`
- **AI for Understanding Content (Embeddings):** OpenAI (`text-embedding-3-small`) via the `openai` package
- **AI for Generating Answers (LLM):** Google Gemini (`1.5 Flash`) via the `google-generativeai` package
- **Vector Database:** ChromaDB via the `chromadb-client` package
- **Project Utilities:** `Pydantic` (for data validation, built into FastAPI), `logging` (Python's standard library), `python-dotenv` (for configuration)

---

### **How It Works: The Two Key Phases**

#### **Phase 1: Adding a Document**

This is how the application "reads" and learns from a PDF. The process is identical in concept.

1.  **Trigger:** You make an API call specifying the name of a PDF file located in a local `files/` folder.
2.  **Parse & Chunk:** The server uses `pypdf` to read the PDF and break its text into small, manageable, and slightly overlapping chunks. It carefully tracks the starting and ending line number for each chunk.
3.  **Embed:** Each text chunk is sent to OpenAI, which converts it into a numerical **embedding**.
4.  **Store:** Each embedding is stored in the Chroma vector database along with its **metadata**: the original filename, page number, and the start/end line numbers.

#### **Phase 2: Asking a Question**

This is how the application uses the learned information to answer your questions.

1.  **Trigger:** You make an API call with your question.
2.  **Embed Question:** The server takes your question and uses OpenAI to convert it into an embedding.
3.  **Find Relevant Context:** The application searches ChromaDB to find the document chunks whose embeddings are most similar to your question's embedding.
4.  **Generate Answer:** The server sends a prompt to Google's Gemini AI. This prompt includes:
    - A clear instruction (e.g., "Answer the user's question based _only_ on the following text").
    - The relevant document chunks it just found.
    - Your original question.
5.  **Respond with Citations:** The server returns Gemini's answer and a list of **citations**—the specific filename and line numbers of the chunks used to generate that answer.

---

### **API Endpoints**

The application will have two simple API routes, defined in FastAPI.

1.  **`POST /embed`**

    - **Purpose:** To process and store a new PDF document.
    - **Request:** `{ "filename": "your-document.pdf" }`
    - **Response:** A confirmation message with the total number of chunks stored.

2.  **`POST /ask`**

    - **Purpose:** To ask a question and get an answer.
    - **Request:** `{ "question": "What is the main topic of the document?" }`
    - **Response:** The answer from the AI and a list of sources (citations).
      ```json
      {
        "answer": "The main topic is...",
        "citations": [
          { "filename": "your-document.pdf", "lineStart": 52, "lineEnd": 65 }
        ]
      }
      ```

---

### **Setup & Configuration**

To run the project, you'll first install the necessary Python packages, likely from a `requirements.txt` file. You'll also need to set the following environment variables:

- `OPENAI_API_KEY`: Your API key for OpenAI.
- `GEMINI_API_KEY`: Your API key for Google Gemini.
- `CHROMA_URL`: The URL for your ChromaDB instance.

You can also adjust these optional settings:

- `CHUNK_SIZE`: How large each text chunk should be.
- `CHUNK_OVERLAP`: How much text the chunks should overlap.
- `TOP_K`: The number of relevant chunks to retrieve for answering a question.
