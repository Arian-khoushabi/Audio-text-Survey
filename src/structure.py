
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent

DATA_DIR = PROJECT_DIR / "data"
MAQA_DIR = DATA_DIR / "mAQA"

SOURCE_METADATA = DATA_DIR / "original.json"
OUTPUT_METADATA = DATA_DIR / "metadata.json"


with open(
    SOURCE_METADATA,
    "r",
    encoding="utf-8"
) as file:

    original_data = json.load(file)


questions = {}


for item in original_data:

    question_text = item["question"].strip()

    if question_text not in questions:

        questions[question_text] = {
            "question_id": f"q_{len(questions)}",
            "question_text": question_text,
            "summary": "",
            "keywords": [],
            "answers": []
        }

    question = questions[question_text]


    for audio_file in item.get(
        "audio_files",
        []
    ):

        audio_path = Path(audio_file)

        filename = audio_path.name

        parts = filename.rsplit(
            "_",
            1
        )

        if (
            len(parts) == 2
            and parts[1].endswith(".wav")
        ):

            number_part = parts[1][:-4]

            if number_part.isdigit():

                corrected_number = (
                    int(number_part) + 1
                )

                filename = (
                    f"{parts[0]}_"
                    f"{corrected_number}.wav"
                )


        relative_audio_path = (
            Path("mAQA")
            / "aud_files"
            / filename
        ).as_posix()


        answer = {

            "answer_id": (
                f"{question['question_id']}"
                f"_a_{len(question['answers'])}"
            ),

            "audio_path": relative_audio_path,

            "transcript": "",

            "status": "pending",

            "emotion": {},

            "emotion_status": "pending"
        }


        question["answers"].append(
            answer
        )


metadata = {

    "questions": list(
        questions.values()
    )
}


with open(
    OUTPUT_METADATA,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        ensure_ascii=False,
        indent=4
    )


print("=" * 60)

print(
    "mAQA metadata created successfully."
)

print("=" * 60)

print(
    f"Questions: "
    f"{len(metadata['questions'])}"
)

print(
    "Total audio files: "
    f"{sum(len(q['answers']) for q in metadata['questions'])}"
)

print("=" * 60)


for question in metadata["questions"]:

    print(
        f"{question['question_id']}: "
        f"{question['question_text']}"
    )

    print(
        f"  Answers: "
        f"{len(question['answers'])}"
    )
