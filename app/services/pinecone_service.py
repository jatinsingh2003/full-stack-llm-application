# import pinecone
# from app.services.openai_service import get_embedding
# import os

# PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')
# EMBEDDING_DIMENSION = 1536

# def embed_chunks_and_upload_to_pinecone(chunks, index_name):
#     if index_name in pinecone.list_indexes():
#         print("\nIndex already exists. Deleting index ...")
#         pinecone.delete_index(name=index_name)
    
#     print("\nCreating a new index: ", index_name)
#     pinecone.create_index(name=index_name,
#                           dimension=EMBEDDING_DIMENSION, metric='cosine')

#     index = pinecone.Index(index_name)

#     # Embedding each chunk and preparing for upload
#     print("\nEmbedding chunks using OpenAI ...")
#     embeddings_with_ids = []
#     for i, chunk in enumerate(chunks):
#         embedding = get_embedding(chunk)
#         embeddings_with_ids.append((str(i), embedding, chunk))

#     print("\nUploading chunks to Pinecone ...")
#     upserts = [(id, vec, {"chunk_text": text}) for id, vec, text in embeddings_with_ids]
#     index.upsert(vectors=upserts)

#     print(f"\nUploaded {len(chunks)} chunks to Pinecone index\n'{index_name}'.")


# def get_most_similar_chunks_for_query(query, index_name):
#     print("\nEmbedding query using OpenAI ...")
#     question_embedding = get_embedding(query)

#     print("\nQuerying Pinecone index ...")
#     index = pinecone.Index(index_name)
#     query_results = index.query(question_embedding, top_k=3, include_metadata=True)
#     context_chunks = [x['metadata']['chunk_text'] for x in query_results['matches']]

#     return context_chunks   


# def delete_index(index_name):
#   if index_name in pinecone.list_indexes():
#     print("\nDeleting index ...")
#     pinecone.delete_index(name=index_name)
#     print(f"Index {index_name} deleted successfully")
#   else:
#      print("\nNo index to delete!")

# app/services/pinecone_service.py
import os
import logging
from typing import List, Dict, Any, Optional

from pinecone import Pinecone, ServerlessSpec

from app.services.openai_service import get_embedding

logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", None)  # optional
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", None)  # e.g. "aws" or "gcp"
PINECONE_REGION = os.getenv("PINECONE_REGION", None)  # e.g. "us-west-2" or "us-central1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "llm-fullstack-index")
EMBEDDING_DIMENSION = int(os.getenv("PINECONE_DIMENSION", 1536))

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY environment variable is not set")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


def _list_indexes() -> List[str]:
    try:
        res = pc.list_indexes()
        if isinstance(res, list):
            return res
        return [idx.name if hasattr(idx, "name") else idx for idx in res]
    except Exception as e:
        logger.error(f"Error listing indexes: {e}")
        return []


def _default_serverless_spec() -> ServerlessSpec:
    """
    Build a ServerlessSpec from env vars or fallbacks.
    """
    cloud = PINECONE_CLOUD
    region = PINECONE_REGION

    # Best-effort defaults if not provided
    if not cloud:
        # heuristic from PINECONE_ENV if present
        if PINECONE_ENV and "gcp" in PINECONE_ENV.lower():
            cloud = "gcp"
        else:
            cloud = "aws"
    if not region:
        region = "us-west-2" if cloud == "aws" else "us-central1"

    return ServerlessSpec(cloud=cloud, region=region)


def embed_chunks_and_upload_to_pinecone(chunks: List[str], index_name: str):
    existing = _list_indexes()
    if index_name in existing:
        print(f"\nIndex '{index_name}' already exists. Deleting index ...")
        pc.delete_index(name=index_name)

    print(f"\nCreating a new index: {index_name}")

    # Build a spec (required by create_index in v7+)
    spec = _default_serverless_spec()

    # Create index (spec is required)
    try:
        pc.create_index(name=index_name, dimension=EMBEDDING_DIMENSION, metric="cosine", spec=spec)
    except TypeError:
        # Some SDK versions require spec positional; try positional fallback
        pc.create_index(index_name, EMBEDDING_DIMENSION, "cosine", spec)

    index = pc.Index(index_name)

    # Embedding each chunk and preparing for upload
    print("\nEmbedding chunks using OpenAI ...")
    vectors = []
    for i, chunk in enumerate(chunks):
        emb = get_embedding(chunk)
        vectors.append({
            "id": str(i),
            "values": emb,
            "metadata": {"chunk_text": chunk}
        })

    print("\nUploading chunks to Pinecone ...")
    resp = index.upsert(vectors=vectors)
    print(f"\nUploaded {len(chunks)} chunks to Pinecone index '{index_name}'.")
    return resp



def _extract_matches_from_query_response(resp: Any) -> List[Dict[str, Any]]:
    """
    Handle possible response shapes from index.query across SDK versions.
    Return list of match dicts that contain 'metadata' keys.
    """
    # Common new form: {'results': [{'matches': [...]}], ...}
    try:
        if isinstance(resp, dict):
            if "results" in resp and isinstance(resp["results"], list):
                first = resp["results"][0]
                if "matches" in first:
                    return first["matches"]
            if "matches" in resp:
                return resp["matches"]
        # Sometimes the response object has attributes
        if hasattr(resp, "results"):
            results = getattr(resp, "results")
            if isinstance(results, list) and len(results) > 0 and hasattr(results[0], "matches"):
                return results[0].matches  # type: ignore
        # As a last resort, try to access resp[0]['matches']
        try:
            return resp[0]["matches"]
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Could not parse Pinecone response shape: {e}")

    # fallback: empty
    return []


def get_most_similar_chunks_for_query(query: str, index_name: str, top_k: int = 3) -> List[str]:
    """
    Embed the query and return top_k most similar chunk_texts from the Pinecone index.
    """
    print("\nEmbedding query using OpenAI ...")
    question_embedding = get_embedding(query)

    print("\nQuerying Pinecone index ...")
    index = pc.Index(index_name)

    # Use the new query signature (queries=list)
    resp = index.query(queries=[question_embedding], top_k=top_k, include_metadata=True)

    matches = _extract_matches_from_query_response(resp)
    # Each match should have metadata with chunk_text
    context_chunks = []
    for m in matches:
        # support both dict access and object access
        meta = None
        if isinstance(m, dict):
            meta = m.get("metadata", {})
        else:
            meta = getattr(m, "metadata", {})  # type: ignore

        chunk_text = None
        if isinstance(meta, dict):
            chunk_text = meta.get("chunk_text")
        else:
            # if metadata is an object with attribute
            chunk_text = getattr(meta, "chunk_text", None)  # type: ignore

        if chunk_text:
            context_chunks.append(chunk_text)

    return context_chunks


def delete_index(index_name: str):
    """Delete index if it exists."""
    existing = _list_indexes()
    if index_name in existing:
        print("\nDeleting index ...")
        pc.delete_index(name=index_name)
        print(f"Index {index_name} deleted successfully")
    else:
        print("\nNo index to delete!")
