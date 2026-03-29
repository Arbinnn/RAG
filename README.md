# RAG (Retrieval-Augmented Generation) System

## What This Repo Does

This is a Retrieval-Augmented Generation system that answers questions about PDF documents. Upload a PDF, and the system automatically chunks the text, generates embeddings, stores them in a vector database, and retrieves relevant passages to generate AI-powered answers to your questions.

### Key Features

- PDF document processing with automatic text chunking (1000 characters with 200 character overlap)
- Vector embeddings using OpenAI text-embedding-3-large model (3072 dimensions)
- Semantic search using Qdrant vector database
- AI-powered answers via GPT-4o-mini through GitHub Models API
- Web interface built with Streamlit
- Asynchronous workflow orchestration using Inngest
- Supports local or remote Qdrant instances

## How to Run It

### Prerequisites

- Python 3.12+
- Node.js (for Inngest CLI)
- GitHub Personal Access Token
- Windows, macOS, or Linux

### Setup Steps

1. **Clone the repository**

   ```powershell
   git clone https://github.com/Arbinnn/RAG.git
   cd RAG
   ```

2. **Create and activate virtual environment**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```powershell
   pip install -e .
   ```

4. **Create .env file** in the project root

   ```
   API_KEY=your_github_pat_token
   OPENAI_BASE_URL=https://models.inference.ai.azure.com/
   ```

   - `API_KEY`: GitHub Personal Access Token from https://github.com/settings/tokens
   - `OPENAI_BASE_URL`: Azure endpoint for GitHub Models (leave as shown)
   - `QDRANT_URL` (optional): Remote Qdrant URL; if not set, uses local embedded storage

5. **Start all services** (use a separate terminal for each)

   **Terminal 1 - Inngest dev server:**

   ```powershell
   npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
   ```

   **Terminal 2 - FastAPI backend:**

   ```powershell
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

   **Terminal 3 - Streamlit frontend:**

   ```powershell
   streamlit run streamlit_app.py
   ```

6. **Access the UI**
   - Opens automatically at http://localhost:8501
   - Upload a PDF and ask questions about it

## Architecture

### Components

- **Streamlit Frontend** (port 8501): Web UI for uploading PDFs and asking questions
- **FastAPI Backend** (port 8000): REST API endpoints for Inngest workflows
- **Inngest Workflow Engine** (port 8288): Orchestrates PDF processing and query workflows
- **Qdrant Vector Database** (port 6333): Stores and retrieves vector embeddings

### Workflows

**PDF Ingestion (`rag/ingest_pdf`)**

1. Load PDF file
2. Split text into chunks (1000 characters, 200 overlap)
3. Generate embeddings for each chunk using OpenAI API
4. Store vectors in Qdrant with source metadata

**Query Processing (`rag/query_pdf_ai`)**

1. Embed user question
2. Search Qdrant for top_k similar chunks
3. Send retrieved context to GPT-4o-mini LLM
4. Return answer with source document name

## Scope

### What It Supports

- Upload and process PDF documents
- Automatic text chunking and embedding
- Semantic similarity search across document content
- AI-powered question answering using retrieved context
- Display answers with source document references

### File Formats

- **PDF**: Text-based PDFs only (not scanned images without OCR)

### Models

- **Embeddings**: OpenAI text-embedding-3-large (3072-dimensional)
- **LLM**: GPT-4o-mini via GitHub Models API
- **Vector DB**: Qdrant v1.17.1+

### Limitations

**File Format Constraints**

- Only PDF documents supported
- One PDF upload at a time
- Image-based or scanned PDFs cannot be processed
- No support for CSV, Excel, Word, or other formats

**System Constraints**

- Chunk size fixed at 1000 characters, 200 character overlap
- Query timeout: 120 seconds
- No user authentication or access control
- No document versioning or deletion
- No cross-document reasoning or multi-hop queries
- No keyword search (semantic search only)

**Operational Constraints**

- Local Qdrant supports single-process access only
- Inngest dev server is for development, not production
- No persistent logging or audit trail

## Usage

### Upload a PDF

1. Go to http://localhost:8501
2. Locate "Upload a PDF to Ingest"
3. Click "Choose a PDF" and select your file
4. Click submit
5. Wait for "Triggered ingestion for: filename.pdf" message

### Ask Questions

1. Scroll to "Ask a question about your PDFs"
2. Type your question (e.g., "What is this document about?")
3. Adjust "How many chunks to retrieve" if needed (default: 5, max: 20)
4. Click "Ask"
5. Wait for answer (2-10 seconds typically)
6. View results with source document name

## Environment Variables

| Variable        | Required | Default                                | Description                                      |
| --------------- | -------- | -------------------------------------- | ------------------------------------------------ |
| API_KEY         | Yes      | -                                      | GitHub Personal Access Token                     |
| OPENAI_BASE_URL | No       | https://models.inference.ai.azure.com/ | Azure endpoint for GitHub Models                 |
| QDRANT_URL      | No       | -                                      | Remote Qdrant server URL; omit for local storage |

## Troubleshooting

### Port Already in Use

Kill lingering processes:

```powershell
Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*node*"} | Stop-Process -Force
```

### Qdrant Storage Already Accessed

Local Qdrant cannot handle multiple processes. Set `QDRANT_URL` in `.env`:

```
QDRANT_URL=http://localhost:6333
```

Then run Qdrant in Docker:

```powershell
docker run -p 6333:6333 qdrant/qdrant:latest
```

### Inngest Won't Connect

1. Verify Inngest running: `npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery`
2. Verify FastAPI running: Check http://127.0.0.1:8000/docs
3. Check terminal output for error messages

### PDF Processing Fails

- Ensure PDF is valid and readable
- Check file size is reasonable (<100 MB)
- Verify API_KEY is correct and has required scopes
- Check logs in FastAPI terminal

### Empty or Irrelevant Answers

- PDF may lack relevant content
- Try different query wording
- Increase top_k to retrieve more chunks (up to 20)

### Query Timeout (120 seconds)

- Qdrant may be slow or unresponsive
- Check network connection
- Try fewer chunks (lower top_k)
- Use remote Qdrant instead of local

## Project Structure

```
.
├── main.py           - FastAPI app with Inngest workflows
├── dataloader.py     - PDF loading and chunking
├── vectorDB.py       - Qdrant storage operations
├── streamlit_app.py  - Streamlit web interface
├── pyproject.toml    - Project dependencies
├── .env              - Environment config (not committed)
├── .gitignore        - Git ignore patterns
└── uploads/          - Directory for uploaded PDFs (created at runtime)
```

## Dependencies

From `pyproject.toml`:

- FastAPI / Uvicorn
- Inngest SDK
- LlamaIndex (document processing)
- OpenAI client
- Qdrant client
- Streamlit
- python-dotenv

## License

This project is for educational and research purposes.
