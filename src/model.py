from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.documents import Document
from collections import defaultdict
import getpass
import os
from pathlib import Path
import json

ROOT_DIR = Path(__file__).parent.parent / "data"
METADATA_FILE = ROOT_DIR / "metadata.json"

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

documents = []

for question in metadata["questions"]:

    for answer in question["answers"]:

        transcript = answer.get("transcript")

        if not transcript:
            continue

        document = Document(
            page_content=transcript,
            metadata={
                "question_id": question["question_id"],
                "answer_id": answer["answer_id"],
                "audio_path": answer["audio_path"],
                "emotion": answer.get("emotion"),
            }
        )

        documents.append(document)


print(f"Created {len(documents)} documents.")

documents_by_question = defaultdict(list)

for document in documents:

    question_id = document.metadata["question_id"]

    documents_by_question[question_id].append(document)

os.environ["COHERE_API_KEY"] = getpass.getpass()
model = ChatCohere(
    model="command-a-plus-05-2026",
    temperature=0.1,
    cohere_api_key=os.environ["COHERE_API_KEY"],
)

system_prompt = "You are a Persian survey analysis assistant" \
"You will receive metadata containing multiple user answers to the same question. Each answer may contain:" \
"- question_id" \
"- transcript" \
"- emotion probabilities" \
"Analyze all answers together and produce ONE overall summary for the question." \
"Requirements:" \
"- Summarize the overall opinions, experiences, positive/negative aspects, problems, suggestions, and recurring themes." \
"- Identify important differences or conflicting opinions when present." \
"- Use emotion probabilities only as supporting context for interpreting the transcriptions." \
"- Do not report emotion probabilities, emotion labels, or emotion scores as separate output unless they are directly relevant to the overall summary." \
"- Never invent information or assume something that is not supported by the answers." \
"- Do not summarize answers individually; combine them into one coherent overall summary." \
"- The summary must be accurate, precise, concise, and natural Persian." \
"- Generate meaningful keywords representing the main topics and recurring themes." \
"- Do not use generic or unsupported keywords." \
"Return ONLY a valid JSON object with exactly this structure:" \
"{" \
'  "summary": "خلاصه کلی پاسخ کاربران",' \
'  "keywords": ["کلیدواژه۱", "کلیدواژه۲", "کلیدواژه۳"]' \
"}" \
"Do not output Markdown, explanations, or anything outside the JSON object."

examples = [
    {
        "input": {
            "question_id": "q1",
            "answers": [
                {
                    "transcript": "I really liked the system.",
                    "emotion": {
                        "neu": 0.10,
                        "hap": 0.85,
                        "ang": 0.02,
                        "sad": 0.03
                    }
                },
                {
                    "transcript": "It was very easy to use.",
                    "emotion": {
                        "neu": 0.20,
                        "hap": 0.75,
                        "ang": 0.03,
                        "sad": 0.02
                    }
                },
                {
                    "transcript": "I had some problems with it.",
                    "emotion": {
                        "neu": 0.40,
                        "hap": 0.10,
                        "ang": 0.35,
                        "sad": 0.15
                    }
                }
            ]
        },
        "output": {
            "summary": "بازخورد کاربران در مجموع مثبت است. کاربران سیستم را دوست‌داشتنی و آسان برای استفاده توصیف کرده‌اند، هرچند یکی از کاربران به وجود برخی مشکلات در استفاده از سیستم اشاره کرده است. احساسات شناسایی‌شده نیز عمدتاً با بازخورد مثبت کاربران همخوانی دارند.",
            "keywords": [
                "رضایت کاربران",
                "سهولت استفاده",
                "مشکلات سیستم"
            ]
        }
    }
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt, 
    examples=examples,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),

        few_shot_prompt,

        (
            "human",
            """
Question ID:
{question_id}

Answers:
{answers}
"""
        ),
    ]
)

chain = prompt | model

for question in metadata["questions"]:

    question_id = question["question_id"]

    question_documents = documents_by_question.get(
        question_id,
        []
    )

    if not question_documents:
        print(
            f"Skipping {question_id}: "
            f"no transcripts available."
        )
        continue

    answers = []

    for document in question_documents:

        answers.append(
            {
                "answer_id": document.metadata["answer_id"],
                "transcript": document.page_content,
                "emotion": document.metadata["emotion"],
            }
        )

    answers_json = json.dumps(
        answers,
        ensure_ascii=False,
        indent=2
    )

    print("\n" + "=" * 70)
    print(f"Processing {question_id}")
    print(f"Number of answers: {len(answers)}")

    response = chain.invoke(
        {
            "question_id": question_id,
            "answers": answers_json,
        }
    )

    try:

        result = json.loads(response.content)

        summary = result["summary"]
        keywords = result["keywords"]

    except (json.JSONDecodeError, KeyError) as error:

        print(
            f"Error parsing response for {question_id}: {error}"
        )

        print("Raw response:")
        print(response.content)

        continue

    question["summary"] = summary
    question["keywords"] = keywords

    print("\nSummary:")
    print(summary)

    print("\nKeywords:")
    print(keywords)

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
        ensure_ascii=False
    )


print("\n" + "=" * 70)
print("All questions processed successfully.")
print(f"Updated metadata saved to: {METADATA_FILE}")