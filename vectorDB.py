from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=3072):
        """
        Initialize connection to Qdrant and ensure the collection exists.

        Args:
            url: Address of the Qdrant server
            collection: Name of the collection (like a table in DB)
            dim: Dimension of embedding vectors (must match your embedding model)
        """

        # Create a client to communicate with Qdrant server
        self.client = QdrantClient(url=url, timeout=30)

        # Store collection name for reuse
        self.collection = collection

        # Check if the collection already exists
        # If not, create it with specified vector configuration
        if not self.client.collection_exists(collection):

            self.client.create_collection(
                collection_name=self.collection,

                # Define how vectors are stored and compared
                vectors_config=VectorParams(
                    size=dim,                   # length of each embedding vector
                    distance=Distance.COSINE    # similarity metric (cosine similarity)
                )
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        """
        Insert or update multiple vectors into Qdrant.

        Args:
            ids: Unique identifiers for each vector
            vectors: List of embedding vectors
            payloads: Metadata for each vector (e.g., text, source)
        """

        # Convert raw data into Qdrant's required format (PointStruct)
        # Each point = (id, vector, metadata)
        points = [
            PointStruct(
                id=ids[i],              # unique ID for the vector
                vector=vectors[i],      # embedding vector
                payload=payloads[i]     # metadata (text, source, etc.)
            )
            for i in range(len(ids))
        ]

        # Insert or update points in the collection
        # If ID exists → update, else → insert
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    def search(self, query_vector: list[float], top_k: int = 5):
        """
        Search for the most similar vectors to a query.

        Args:
            query_vector: Embedding of the user's query
            top_k: Number of top similar results to return

        Returns:
            Dictionary containing:
                - contexts: list of retrieved text chunks
                - sources: set of unique sources
        """

        # Perform similarity search in Qdrant
        results = self.client.search(
            collection_name=self.collection,   # where to search
            query_vector=query_vector,         # query embedding
            with_payload=True,                 # include metadata in results
            limit=top_k                        # number of results to return
        )

        # Store extracted text and sources
        contexts = []   # relevant text chunks
        sources = set() # unique source references (no duplicates)

        # Process each search result
        for r in results:

            # Safely extract payload (metadata)
            # If payload doesn't exist → use empty dict
            payload = getattr(r, "payload", None) or {}

            # Extract stored text and source info
            text = payload.get("text", "")
            source = payload.get("source", "")

            # Only keep valid text entries
            if text:
                contexts.append(text)
                sources.add(source)

        # Return structured result for RAG pipeline
        return {
            "contexts": contexts,   # list of relevant text chunks
            "sources": sources      # unique sources (e.g., file names)
        }