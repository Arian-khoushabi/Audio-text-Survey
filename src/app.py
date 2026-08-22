import streamlit as st
from src.main import LLM_Chat
from pathlib import Path
import json
from dotenv import load_dotenv
import os


@st.cache_resource
def get_backend(api_key):
    return LLM_Chat(api_key)


class App:
    def __init__(self):
        st.set_page_config(
            page_title="Langchain: Chat With Cohere",
            page_icon="content/logo.png"
        )

        st.title("ُUser chatbot: Chat With Cohere")

        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');

                * {
                    font-family: 'Vazirmatn', sans-serif;
                }
            </style>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        load_dotenv()
        default_key = os.getenv("COHERE_API_KEY", "")
        
        api_key = st.sidebar.text_input(
        "Cohere API Key", 
        value=default_key,
        type="password"
        )

        if not api_key:
            st.info("Please enter Cohere API key to continue.")
            st.stop()

        os.environ["COHERE_API_KEY"] = api_key
        return api_key

    def display_message(self, message_type, content):
        if message_type == "human":
            st.chat_message("human").write(content)

        else:
            with st.chat_message("ai"):
                answer_stream = (
                    chunk.get("answer", "")
                    for chunk in content
                    if isinstance(chunk, dict)
                )
                st.write_stream(answer_stream)

    def get_user_input(self):
        return st.chat_input("سوال خود را بپرسید...")

    def display_rag_tab(self):
        api_key = self.display_sidebar()

        os.environ["COHERE_API_KEY"] = api_key

        backend = get_backend(api_key)

        if not backend.get_chat_history():
            backend.reset_chat()

        for message in backend.get_chat_history():
            if message.type == "human":
                st.chat_message("human").write(message.content)

            elif message.type == "ai":
                st.chat_message("ai").write(message.content)

        prompt = self.get_user_input()

        if prompt:
            self.display_message("human", prompt)

            response = backend.process_input(prompt)

            self.display_message("ai", response)

    def load_metadata(self):

        root_dir = Path(__file__).parent.parent / "data"
        metadata_file = root_dir / "metadata.json"

        if not metadata_file.exists():
            st.error(
                f"metadata.json was not found at:\n{metadata_file}"
            )
            return None

        try:
            with open(
                metadata_file,
                "r",
                encoding="utf-8"
            ) as file:
                return json.load(file)

        except Exception as error:
            st.error(
                f"Could not load metadata.json: {error}"
            )
            return None

    def display_summary(self, question):
        question_text = question.get(
            "question_text",
            "سوال موجود نیست."
        )

        summary = question.get(
            "summary",
            "برای این سوال هنوز خلاصه‌ای تولید نشده است."
        )

        keywords = question.get(
            "keywords",
            []
        )

        st.markdown("---")

        st.subheader("سوال")
        st.write(question_text)

        st.subheader("خلاصه")
        st.write(summary)

        st.subheader("کلمات کلیدی")

        if keywords:
            keyword_columns = st.columns(
                min(len(keywords), 4)
            )

            for index, keyword in enumerate(keywords):
                with keyword_columns[index % len(keyword_columns)]:
                    st.info(keyword)

        else:
            st.write(
                "کلمه کلیدی‌ای برای این سوال ثبت نشده است."
            )

    def display_summary_tab(self):
        st.header("📊 خلاصه پاسخ‌های کاربران")

        metadata = self.load_metadata()

        if metadata is None:
            return

        questions = metadata.get(
            "questions",
            []
        )

        if not questions:
            st.warning(
                "هیچ سوالی در metadata.json پیدا نشد."
            )
            return

        st.write(
            "برای مشاهده خلاصه و کلمات کلیدی، "
            "یکی از سوالات را انتخاب کنید."
        )

        if "selected_question_id" not in st.session_state:
            st.session_state.selected_question_id = None

        number_of_columns = 3

        columns = st.columns(number_of_columns)

        for index, question in enumerate(questions):

            question_id = question.get(
                "question_id",
                str(index)
            )

            question_text = question.get(
                "question_text",
                f"Question {index + 1}"
            )

            button_text = question_text

            if len(button_text) > 70:
                button_text = (
                    button_text[:67] + "..."
                )

            with columns[index % number_of_columns]:

                if st.button(
                    button_text,
                    key=f"question_button_{question_id}",
                    use_container_width=True
                ):
                    st.session_state.selected_question_id = (
                        question_id
                    )

        selected_question = None

        for question in questions:
            if (
                question.get("question_id")
                == st.session_state.selected_question_id
            ):
                selected_question = question
                break

        if selected_question is not None:
            self.display_summary(
                selected_question
            )

    def display_app(self):

        tab_rag, tab_summary = st.tabs(
            [
                "💬 RAG Chat",
                "📊 Summarization"
            ]
        )

        with tab_rag:
            self.display_rag_tab()

        with tab_summary:
            self.display_summary_tab()


if __name__ == "__main__":
    app = App()
    app.display_app()