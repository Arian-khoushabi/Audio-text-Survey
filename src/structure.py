import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent / "data"

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}

for question_folder in sorted(ROOT_DIR.iterdir()):

    if not question_folder.is_dir():
        continue

    metadata = []

    for audio_file in sorted(question_folder.iterdir()):

        if audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        metadata.append({
            "question": question_folder.name,
            "answer": audio_file.name,
            "audio_path": str(audio_file.relative_to(ROOT_DIR)),
            "transcript": "",
            "emotion": None,
            "summary": None,
            "status": "pending"
        })

    metadata_file = question_folder / "metadata.json"

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

print("Metadata files created successfully.")