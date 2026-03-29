from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv()

EMBED_MODEL = "text-embedding-3-large"
EMBED_MODEL_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("API_KEY") 
    base_url = os.getenv("OPENAI_BASE_URL")

    # GitHub-issued keys in API_KEY must use the GitHub Models endpoint.
    if not base_url and isinstance(api_key, str) and api_key.startswith("github_"):
        base_url = "https://models.inference.ai.azure.com"

    # GitHub Models tokens use an OpenAI-compatible endpoint.
    if not base_url and os.getenv("GITHUB_TOKEN") and not os.getenv("API_KEY"):
        base_url = "https://models.inference.ai.azure.com"

    if not api_key:
        raise ValueError("Missing API key. Set API_KEY or GITHUB_TOKEN in your environment.")

    return OpenAI(api_key=api_key, base_url=base_url)

def load_and_split_pdf(file_path: str) -> list[str]:
    """
    Load a PDF file and split it into text chunks.

    Args:
        file_path: Path to the PDF file
    Returns:
        List of text chunks extracted from the PDF  
    """
    docs = SimpleDirectoryReader(input_files=[Path(file_path)]).load_data()
    texts = [doc.text for doc in docs if getattr(doc, "text", None) is not None]
    chunks: list[str] = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks.

    Args:
        texts: List of text chunks to embed 
    Returns:
        List of embedding vectors corresponding to the input texts
    """
    clean_texts = [text.strip() for text in texts if isinstance(text, str) and text.strip()]
    if not clean_texts:
        raise ValueError("embed_texts received no valid non-empty strings.")

    client = _get_openai_client()
    response = client.embeddings.create(
        input=clean_texts,
        model=EMBED_MODEL
    )
    return [data.embedding for data in response.data]