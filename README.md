# Audio Survey Analysis

This project processes audio responses collected from a survey.

Each survey question has its own folder, and the audio responses to that question are stored inside the corresponding folder. The project creates metadata for the audio files and prepares the data for future speech recognition, text analysis, summarization, and emotion analysis.

## Project Goals

The main goals of this project are:

1. Organize survey audio responses.
2. Create metadata for questions and answers.
3. Convert audio responses to text using Automatic Speech Recognition (ASR).
4. Analyze and summarize the responses using a Large Language Model (LLM).
5. Detect emotions expressed in the audio responses.

## Project Structure

```text
Survey/
│
├── data/
│   ├── question_1/
│   │   ├── audio_1.wav
│   │   ├── audio_2.wav
│   │   └── audio_3.wav
│   │
│   ├── question_2/
│   │   ├── audio_1.wav
│   │   └── audio_2.wav
│   │
│   └── ...
│
├── src/
│   └── structure.py
│
├── README.md
└── requirements.txt