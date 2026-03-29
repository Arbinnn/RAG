from fastapi import FastAPI
import inngest
import logging
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
from inngest.experimental import ai
from dataloader import load_and_split_pdf, embed_texts
from vectorDB import QdrantStorage

load_dotenv()


def _resolve_llm_auth() -> tuple[str, str | None]:
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not base_url and isinstance(api_key, str) and api_key.startswith("github_"):
        base_url = "https://models.inference.ai.azure.com"

    if not api_key:
        raise ValueError("Missing API key. Set API_KEY, OPENAI_API_KEY, or GITHUB_TOKEN.")

    return api_key, base_url

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG:Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def ingest_pdf(ctx: inngest.Context):
    def _load_and_chunk(ctx: inngest.Context) -> dict[str, list[str] | str]:
        pdf_path = ctx.event.data.get("pdf_path")
        if not isinstance(pdf_path, str) or not pdf_path:
            raise ValueError("Event data must include a non-empty string 'pdf_path'.")

        source_id = ctx.event.data.get("source_id", pdf_path)
        if source_id is None:
            source_id = pdf_path

        raw_chunks = load_and_split_pdf(pdf_path)
        chunks = [str(chunk).strip() for chunk in raw_chunks if chunk is not None and str(chunk).strip()]
        source_id = str(source_id)

        return {"chunks": chunks, "source_id": source_id}

    def _ingested(load_result: dict[str, list[str] | str]) -> dict[str, int]:
        chunks = load_result.get("chunks", [])
        source_id = str(load_result.get("source_id", "unknown"))
        if not isinstance(chunks, list):
            raise ValueError("Step result must include chunks as a list.")
        chunks = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]

        if not chunks:
            return {"ingested": 0}

        vectors = embed_texts(chunks)
        if len(vectors) != len(chunks):
            raise ValueError("Embedding count does not match chunk count.")

        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
        payloads = [{"text": chunks[i], "source": source_id} for i in range(len(chunks))]
        qdrant = QdrantStorage(collection="rag_collection")
        qdrant.upsert(ids=ids, vectors=vectors, payloads=payloads)

        return {"ingested": len(chunks)}

    load_result = await ctx.step.run("load-and-chunk", lambda: _load_and_chunk(ctx))
    ingested = await ctx.step.run("ingested", lambda: _ingested(load_result))
    return ingested

#QueryPDF
@inngest_client.create_function(
    fn_id="RAG:Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> dict[str, list[str]]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Event data must include a non-empty string 'question'.")

        question = question.strip()
        query_vector = embed_texts([question])[0]
        qdrant = QdrantStorage(collection="rag_collection")
        search_results = qdrant.search(query_vector=query_vector, top_k=top_k)
        return {
            "contexts": list(search_results["contexts"]),
            "sources": list(search_results["sources"]),
        }
    
    question = ctx.event.data.get("question")
    top_k = int(ctx.event.data.get("top_k", 5))
    if top_k < 1:
        top_k = 1

    search_results = await ctx.step.run("search", lambda: _search(question, top_k))

    context_block = "\n\n".join(search_results["contexts"])
    user_content = (
        "Use the following retrieved contexts to answer the question. If you don't know the answer, say you don't know. Always use all available information from the retrieved contexts.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Contexts:\n{context_block}\n\n"
        "Answer consiely based on the retrieved contexts:"
    )

    llm_api_key, llm_base_url = _resolve_llm_auth()
    adapter = ai.openai.Adapter(
        auth_key=llm_api_key,
        base_url=llm_base_url,
        model="gpt-4o-mini",
    )

    res= await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant for answering questions based on retrieved document contexts."},
                {"role": "user", "content": user_content}
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
        }
    )

    answer=res.get("choices", [{}])[0].get("message", {}).get("content", "No answer generated.")
    return {"answer": answer, "sources": search_results["sources"], "num_contexts": len(search_results["contexts"])}


app=FastAPI()

inngest.fast_api.serve(app, inngest_client,[ingest_pdf, query_pdf_ai])