from collections import defaultdict
from pathlib import Path
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.documentize import load_documents

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

documents = load_documents()

print(
    f"Created {len(documents)} documents."
)

documents_by_question = defaultdict(list)

for document in documents:

    question_id = document.metadata["question_id"]

    documents_by_question[question_id].append(
        document
    )


print(
    f"Questions indexed: "
    f"{len(documents_by_question)}"
)

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",

    model_kwargs={
        "device": "cpu",
    },

    encode_kwargs={
        "normalize_embeddings": True,
    },
)

print("\nCreating Chroma vector database...")

ids = [
    f"{document.metadata['question_id']}_{document.metadata['answer_id']}"
    for document in documents
]

vector_db = Chroma.from_documents(
    documents=documents,
    ids=ids,
    embedding=embeddings,
    persist_directory=str(CHROMA_DIR),
    collection_name="survey_answers",
)


print(
    f"Chroma database created at: "
    f"{CHROMA_DIR}"
)



print("\nEmbedding completed.")

print(
    "\nYou now have:"
    "\n- Chroma vector database"
)