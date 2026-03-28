from fastapi import FastAPI
import inngest
import logging
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime
from inngest.experimental import ai

load_dotenv()

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
    return {"message": "PDF ingested successfully!"}

app=FastAPI()

inngest.fast_api.serve(app, inngest_client,[ingest_pdf])