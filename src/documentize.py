from pathlib import Path
import json
from langchain_core.documents import Document


def load_documents():

    ROOT_DIR = Path(__file__).parent.parent
    METADATA_FILE = ROOT_DIR / "data" / "metadata.json"

    with open(METADATA_FILE, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    documents = []

    for question in metadata["questions"]:

        question_id = question["question_id"]

        question_text = question.get(
            "question_text", ""
        ).strip()

        if not question_text:
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
    answer.get("emotion", {}),
    ensure_ascii=False
),
                    "summary": question.get("summary", ""),
                    "keywords": question.get("keywords", []),
                },
            )

            documents.append(document)

    return documents