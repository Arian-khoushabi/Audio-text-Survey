import json
from pathlib import Path

import whisper
import torch


PROJECT_DIR = Path(__file__).parent.parent

DATA_DIR = PROJECT_DIR / "data"

METADATA_PATH = DATA_DIR / "metadata.json"

MODEL_NAME = "large-v3"

LANGUAGE = "English"


def load_metadata():
    """خواندن metadata.json"""

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_metadata(metadata):
    """ذخیره اطلاعات جدید در metadata.json"""

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=4
        )


def main():

    metadata = load_metadata()

    print(
        f"Loading Whisper model: {MODEL_NAME}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"Using device: {device}"
    )

    if device == "cuda":
        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    model = whisper.load_model(
        MODEL_NAME,
        device=device
    )

    print(
        "Whisper model loaded."
    )

    print("-" * 60)

    for question in metadata["questions"]:

        print(
            f"Question: "
            f"{question['question_text']}"
        )

        for answer in question["answers"]:

            answer_id = answer["answer_id"]

            if answer["status"] == "completed":

                print(
                    f"{answer_id}: "
                    "Already processed. Skipping."
                )

                continue

            audio_path = (
                DATA_DIR /
                answer["audio_path"]
            )

            if not audio_path.exists():

                print(
                    f"{answer_id}: "
                    f"Audio file not found."
                )

                answer["status"] = "failed"

                answer["error"] = (
                    f"File not found: "
                    f"{audio_path}"
                )

                save_metadata(metadata)

                continue

            print(
                f"{answer_id}: "
                f"Processing..."
            )

            try:

                result = model.transcribe(
                    str(audio_path),
                    language=LANGUAGE,
                    task="transcribe",
                    fp16=(device == "cuda")
                )

                answer["transcript"] = (
                    result["text"].strip()
                )

                answer["status"] = "completed"

                answer["transcription_model"] = (
                    MODEL_NAME
                )

                answer.pop(
                    "error",
                    None
                )

                save_metadata(metadata)

                print(
                    f"{answer_id}: "
                    "Completed."
                )

                print(
                    f"Transcript: "
                    f"{answer['transcript']}"
                )

            except Exception as error:

                print(
                    f"{answer_id}: "
                    f"Failed."
                )

                print(
                    f"Error: {error}"
                )

                answer["status"] = "failed"

                answer["error"] = str(
                    error
                )

                save_metadata(metadata)

        print("-" * 60)

    print(
        "All audio files have been processed."
    )


if __name__ == "__main__":
    main()