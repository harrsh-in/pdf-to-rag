# Project Overview – RAG MVP Demo

## Objective

Demonstrate a full Retrieval‑Augmented Generation (RAG) loop on Node.js with TypeScript, integrating OpenAI for embeddings, Chroma as the vector database, and Gemini for answer generation. The focus is on clarity of data flow and minimal surface area: one embed route, one question‑answer route, plus line‑level citations.

## Key Technologies

- **Runtime & Language**: Node.js LTS with TypeScript
- **Web Framework**: Express
- **PDF Parsing**: pdf‑parse (native MuPDF bindings under the hood)
- **Embeddings Provider**: OpenAI text‑embedding‑3‑small
- **Vector Store**: Managed cloud Chroma DB
- **LLM Provider**: Google Gemini 1.5 Flash
- **Validation & Utilities**: Zod for schema validation, pino for logging, dotenv for configuration

## MVP Feature Set

1. Upload‑by‑reference: client names a PDF already present in the "files" folder in the root directory; server embeds it.
2. Similarity search: question is embedded, matched against stored vectors, and ranked top‑k.
3. Contextual answering: Gemini receives retrieved chunks as context and produces an answer.
4. Citations: response includes filename and absolute line numbers for every supporting chunk.

## High‑Level Workflow

1. **Embed phase**

   1. Client calls POST /embed with a filename.
   2. Server resolves the path within files, parses text page by page, splits into overlapping chunks, and counts line numbers.
   3. Chunks are embedded in batches via OpenAI and upserted into a dedicated Chroma collection together with metadata (filename, page, lineStart, lineEnd).

2. **Question‑answer phase**

   1. Client calls POST /ask with a free‑form question.
   2. Server embeds the question, runs a similarity query (cosine + top‑k), and gathers the best chunks.
   3. Gemini receives a prompt consisting of a concise system instruction, the retrieved chunks, and the user question.
   4. Gemini’s completion is returned verbatim; citations are mapped from the retrieved metadata and included in the JSON response.

## API Contract

### POST /embed

- **Request body**: object with a single string field called filename.
- **Success response**: JSON containing status ok and the number of chunks stored.
- **Failure cases**: 400 for missing file or validation error; 500 for unexpected server errors.

### POST /ask

- **Request body**: object with a single string field called question.
- **Success response**: JSON with answer text and an array of citation objects (filename, lineStart, lineEnd).
- **Failure cases**: 400 for empty question; 500 for embedding, retrieval, or LLM failures.

## Data Model in Chroma

Each vector upsert includes:

- **id** – SHA‑256 hash of the chunk text (deduplication key)
- **embedding** – 1536‑dimensional float array
- **metadata**

  - filename (string)
  - page (integer)
  - lineStart (integer)
  - lineEnd (integer)

## Configuration & Environment Variables

- OPENAI_API_KEY – required
- GEMINI_API_KEY – required
- CHROMA_URL – root endpoint of the Chroma instance
- CHUNK_SIZE, CHUNK_OVERLAP, TOP_K – optional tuning knobs with safe defaults

## Deployment & Ops

- Single‑binary Node application packaged with esbuild and pino‑pretty for local runs.
- Chroma can run in‑process via chroma run‑server for demo or point to a managed instance.
- Logging includes request ID, route, latency, OpenAI token count, and Gemini token count.

## MVP Limitations

- No streaming of Gemini tokens (plain JSON response).
- PDFs must already exist in files directory; no upload endpoint.
- No automatic ingestion trigger on startup.
- No authentication or rate limiting.
