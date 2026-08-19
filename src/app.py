import streamlit as st
from src.main import LLM_Chat


@st.cache_resource
def get_backend(api_key):
    return LLM_Chat(api_key)


class App:
    def __init__(self):
        st.set_page_config(
            page_title="Langchain: Chat With Cohere",
            page_icon="content/logo.png"
        )

        st.title("🦜 Langchain: Chat With Cohere")

        st.markdown("""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500&display=swap');
                * {
                    font-family: 'Vazirmatn', sans-serif;
                }
            </style>
        """, unsafe_allow_html=True)

    def display_sidebar(self):
        api_key = st.sidebar.text_input(
            "Cohere API Key",
            type="password"
        )

        if not api_key:
            st.info("Please enter Cohere API key to continue.")
            st.stop()

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

    def display_app(self):
        api_key = self.display_sidebar()

        # IMPORTANT:
        # This will only create LLM_Chat once for this API key.
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


if __name__ == "__main__":
    app = App()
    app.display_app()