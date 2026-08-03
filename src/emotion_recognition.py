import json
from pathlib import Path

from transformers import pipeline


PROJECT_DIR = Path(__file__).parent.parent

DATA_DIR = PROJECT_DIR / "data"

METADATA_PATH = DATA_DIR / "metadata.json"


MODEL_NAME = "superb/wav2vec2-base-superb-er"



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
        f"Loading Emotion model: {MODEL_NAME}"
    )


    emotion_model = pipeline(
        "audio-classification",
        model=MODEL_NAME
    )


    print(
        "Emotion model loaded."
    )


    print("-" * 60)



    for question in metadata["questions"]:


        print(
            f"Question: "
            f"{question['question_text']}"
        )


        for answer in question["answers"]:


            answer_id = answer["answer_id"]


            # Skip completed emotion files
            if answer.get("emotion_status") == "done":

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
                    "Audio file not found."
                )


                answer["emotion_status"] = "failed"

                answer["emotion_error"] = (
                    f"File not found: "
                    f"{audio_path}"
                )


                save_metadata(metadata)

                continue



            print(
                f"{answer_id}: "
                "Processing..."
            )



            try:


                result = emotion_model(
                    str(audio_path)
                )


                emotions = {}


                for item in result:

                    emotions[item["label"]] = round(
                        item["score"],
                        4
                    )



                answer["emotion"] = emotions


                answer["emotion_status"] = "done"


                answer["emotion_model"] = (
                    MODEL_NAME
                )


                answer.pop(
                    "emotion_error",
                    None
                )


                save_metadata(metadata)



                print(
                    f"{answer_id}: "
                    "Completed."
                )


                print(
                    f"Emotion: "
                    f"{answer['emotion']}"
                )



            except Exception as error:


                print(
                    f"{answer_id}: "
                    "Failed."
                )


                print(
                    f"Error: {error}"
                )


                answer["emotion_status"] = "failed"


                answer["emotion_error"] = str(
                    error
                )


                save_metadata(metadata)



        print("-" * 60)



    print(
        "All audio files have been processed."
    )



if __name__ == "__main__":
    main()