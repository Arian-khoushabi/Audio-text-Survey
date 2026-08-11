from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

system_prompt = """
You are a careful Persian survey analysis assistant.

You will receive multiple answers given by users to the same survey question.
Each answer contains:
- an answer_id
- a transcript
- emotion probabilities

Your task is to analyze ALL answers together and produce ONE overall
summary of the responses.

IMPORTANT PRINCIPLES:

1. Base the analysis primarily on the transcripts.
   The transcript is the primary source of information.

2. Use emotion probabilities only as supporting information.
   Do not treat an emotion probability as evidence of a user's opinion,
    preference, intention, or evaluation.
    If the transcripts themselves do not explicitly express an emotion or opinion,
    do not claim that they do.
    When relevant, you may mention that the emotional tone varies across answers,
    but do not infer the cause or meaning of that variation.

3. Do not invent information.
   Do not infer opinions, intentions, causes, preferences, or meanings that are
   not supported by the transcripts.

4. Do not assume that an emotion means a specific opinion.
   For example, anger does not necessarily mean that the user disliked the
   subject mentioned in the transcript.

5. Do not infer emotion from punctuation, capitalization, wording, or symbols.
   Use only the provided emotion probabilities when discussing emotional tone.

6. Distinguish between:
   - what users explicitly said,
   - recurring patterns in the answers,
   - differences or contradictions between answers.

7. If the answers are descriptive rather than opinion-based, say so.
   Do not force the analysis into opinions, positive/negative aspects,
   problems, or suggestions when these are not present.

8. Do not summarize each answer separately.
   Combine the answers into one coherent overall summary.

9. When a pattern appears frequently, mention it as a recurring pattern.
   Do not claim exact proportions or percentages unless they can be reliably
   determined from the provided answers.

10. Do not make claims about which group is larger unless the provided answers
    clearly support that conclusion.

11. Keep the summary concise, factual, and natural Persian.

12. Generate a small number of meaningful keywords.
    Keywords must be directly supported by the transcripts and represent the
    main recurring topics or themes.
    Avoid generic keywords such as "پاسخ کاربران", "احساسات", or "نظرات"
    unless they are genuinely meaningful in the context.

OUTPUT:

Return a JSON object containing exactly two fields:

- "summary": a concise overall summary in natural Persian
- "keywords": a list of meaningful Persian keywords

Do not include explanations outside the JSON object.
"""

examples = [
    {
        "input": json.dumps(
            {
                "question_id": "q_example",
                "answers": [
                    {
                        "transcript": "بچه‌ها کنار در صحبت می‌کنند.",
                        "emotion": {
                            "neu": 0.70,
                            "hap": 0.20,
                            "ang": 0.05,
                            "sad": 0.05
                        }
                    },
                    {
                        "transcript": "سگ‌ها کنار در نشسته‌اند.",
                        "emotion": {
                            "neu": 0.80,
                            "hap": 0.10,
                            "ang": 0.05,
                            "sad": 0.05
                        }
                    }
                ]
            },
            ensure_ascii=False,
            indent=2
        ),
        "output": json.dumps(
            {
                "summary": "پاسخ‌ها عمدتاً دو صحنه را توصیف می‌کنند: بچه‌ها در حال صحبت در کنار در و سگ‌ها در حال نشستن در کنار در. پاسخ‌ها بیشتر توصیفی هستند و نظر یا ارزیابی صریحی ارائه نمی‌کنند.",
                "keywords": [
                    "بچه‌ها",
                    "سگ‌ها",
                    "کنار در"
                ]
            },
            ensure_ascii=False,
            indent=2
        )
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

parser = JsonOutputParser()

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

{format_instructions}
"""
        ),
    ]
)

chain = prompt | model | parser

for question in metadata["questions"]:

    question_id = question["question_id"]

    if question.get("summary") and question.get("keywords"):
        print(f"Skipping {question_id}: already processed.")
        continue

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

    max_retries = 2

    for attempt in range(1, max_retries + 1):

        try:

            response = chain.invoke(
                {
                    "question_id": question_id,
                    "answers": answers_json,
                    "format_instructions": parser.get_format_instructions(),
                }
            )

            summary = response.get("summary")
            keywords = response.get("keywords")

            if not isinstance(summary, str) or not isinstance(keywords, list):
                raise ValueError("Invalid format received from model.")

        except Exception as error:

            print(
                f"Attempt {attempt}/{max_retries} failed "
                f"for {question_id}: {error}"
            )

            if attempt == max_retries:

                print(
                    f"Failed to process {question_id} "
                    f"after {max_retries} attempts."
                )

                summary = None
                keywords = None


    if summary is None or keywords is None:
        continue

    question["summary"] = summary
    question["keywords"] = keywords

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

    print("\nSummary:")
    print(summary)

    print("\nKeywords:")
    print(keywords)

    print(f"\n{question_id} saved successfully.")
