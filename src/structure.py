import json
from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent / "data"

AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg"
}

questions = []

for question_number, question_folder in enumerate(
    sorted(ROOT_DIR.iterdir()),
    start=1
):

    if not question_folder.is_dir():
        continue

    answers = []

    for answer_number, audio_file in enumerate(
        sorted(question_folder.iterdir()),
        start=1
    ):

        if not audio_file.is_file():
            continue

        if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        answers.append({
            "answer_id": f"q{question_number}_a{answer_number}",
            "audio_path": str(
                audio_file.relative_to(ROOT_DIR)
            ),
            "transcript": "",
            "emotion": None,
            "status": "pending"
        })

    questions.append({
        "question_id": f"q{question_number}",
        "question_text": question_folder.name,
        "answers": answers,
        "summary": None
    })

metadata = {
    "questions": questions
}

metadata_file = ROOT_DIR / "metadata.json"

with open(
    metadata_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


print(
    f"Metadata file created successfully:\n{metadata_file}"
)