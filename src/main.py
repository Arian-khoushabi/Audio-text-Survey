from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CohereRerank
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from src.retriever import reranked_reteriver 
from pathlib import Path
import streamlit as st
import getpass
import json
import os

class LLM_Chat:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatCohere(
            model="command-a-plus-05-2026",
            temperature=0.0,
            cohere_api_key=self.api_key,
            streaming=True
        )
        self.chat_history = StreamlitChatMessageHistory(key="special_app_key")


        memory_system_prompt = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is."""

        memory_q_prompt = ChatPromptTemplate.from_messages([
            ("system", memory_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        reteriver = reranked_reteriver()
        history_aware_retriever = create_history_aware_retriever(
            self.llm, reteriver, memory_q_prompt
        )

        qa_system_prompt = """You are an assistant for a QA system that answers questions based on responses provided by our users.

        Use the retrieved context below to answer the user's question.

        Rules:
        - Use only the information provided in the retrieved context.
        - If the context does not contain enough information to answer the question, say that you don't know.
        - The context may contain answers related to different survey questions. Combine relevant information from multiple responses when necessary.
        - Consider the survey question associated with each answer when interpreting the response.
        - Do not invent or assume information that is not supported by the context.
        - Keep the answer concise, clear, and natural.
        - Return the answer as plain text, not JSON.
        - Your answers must be persian.

        Retrieved context:
        {context}
        """

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        self.chain_with_message_history = RunnableWithMessageHistory(
            self.rag_chain,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

    def reset_chat(self):
        self.chat_history.clear()
        self.chat_history.add_ai_message("سلام، برای پیدا کردن جواب سوالات شما آماده‌ام.")

    def get_chat_history(self):
        return self.chat_history.messages

    def process_input(self, prompt):
        config = {"configurable": {"session_id": "any"}}
        try:
            response = self.chain_with_message_history.stream({"input": prompt}, config)
            return response
        except Exception as error:
            return str(error)