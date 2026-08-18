from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CohereRerank
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.documentize import load_documents
from pathlib import Path
import json
import os

def reranked_reteriver():

    ROOT_DIR = Path(__file__).parent.parent
    CHROMA_DIR = ROOT_DIR / "chroma_db"
    
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }

    )

    vector_db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="survey_answers",
    )

    vector_retriever = vector_db.as_retriever(
        search_type="mmr",

        search_kwargs={
            "k": 20,
            "lambda_mult": 0.5,
        },
    )

    print("\nCreating BM25 retriever...")

    bm25_retriever = BM25Retriever.from_documents(
        documents
    )

    bm25_retriever.k = 20

    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever,
        ],

        weights=[
            0.5,
            0.5,
        ],
    )

    reranker = CohereRerank(
        model="rerank-multilingual-v3.0", 
        top_n= 5,
        cohere_api_key=os.environ["COHERE_API_KEY"]
    )

    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

    return reranked_retriever