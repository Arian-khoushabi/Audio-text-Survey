from collections import defaultdict
from pathlib import Path
import json
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

ROOT_DIR = Path(__file__).parent.parent / "data"
METADATA_FILE = ROOT_DIR / "metadata.json"

CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

documents = []

for question in metadata["questions"]:

    question_id = question["question_id"]

    question_text = question.get(
        "question_text",
        ""
    ).strip()

    if not question_text:
        print(
            f"Skipping {question_id}: "
            "question_text is missing."
        )
        continue

    for answer in question.get("answers", []):

        transcript = answer.get("transcript")

        if not transcript:
            continue

        document = Document(
            page_content=f"""
Question:
{question_text}

Answer:
{transcript}
""".strip(),

            metadata={
                "question_id": question_id,
                "question_text": question_text,
                "answer_id": answer["answer_id"],
                "audio_path": answer.get("audio_path"),
                "emotion": json.dumps(
                    answer.get("emotion"),
                    ensure_ascii=False,
                ),

                "summary": question.get("summary", ""),
                "keywords": json.dumps(
                    question.get("keywords", []),
                    ensure_ascii=False,
                ),
            },
        )

        documents.append(document)


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

vector_db = Chroma.from_documents(
    documents=documents,

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