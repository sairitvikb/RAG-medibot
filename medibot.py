import os
import time
import json
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# ----------------------------
# 1️⃣ Load FAISS Vectorstore
# ----------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


# ----------------------------
# 2️⃣ Load LLM (Flan-T5)
# ----------------------------
def load_llm():
    flan_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=-1  # CPU (-1) | GPU (0)
    )
    return HuggingFacePipeline(pipeline=flan_pipeline)


# ----------------------------
# 3️⃣ Main Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="🩺 MediBot", page_icon="💬", layout="wide")

    st.title("🩺 MediBot - Your Medical Assistant")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "feedback" not in st.session_state:
        st.session_state.feedback = {}

    # Sidebar controls
    with st.sidebar:
        st.subheader("⚙️ Controls")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.feedback = {}
            st.rerun()

        if st.session_state.messages:
            st.download_button(
                "📥 Download Conversation",
                data=json.dumps(st.session_state.messages, indent=2),
                file_name="conversation.json",
                mime="application/json",
            )

    # Display previous messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Feedback buttons only for bot responses
            if message["role"] == "assistant":
                col1, col2 = st.columns([0.1, 0.9])
                with col1:
                    if st.button("👍", key=f"like_{i}"):
                        st.session_state.feedback[i] = "like"
                        st.success("You liked this response!")
                with col2:
                    if st.button("👎", key=f"dislike_{i}"):
                        st.session_state.feedback[i] = "dislike"
                        st.error("You disliked this response!")

    # Chat input
    user_input = st.chat_input("Ask MediBot something...")
    if user_input:
        # Save and display user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Process response
        with st.spinner("MediBot is thinking..."):
            try:
                vectorstore = get_vectorstore()
                llm = load_llm()

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=False,
                )

                response = qa_chain.invoke({"query": user_input})
                answer = response.get("result", "⚠️ I couldn’t find an answer.")

            except Exception as e:
                answer = f"⚠️ Error: {str(e)}"

        # Typing effect
        with st.chat_message("assistant"):
            placeholder = st.empty()
            typed_text = ""
            for char in answer:
                typed_text += char
                placeholder.markdown(typed_text)
                time.sleep(0.02)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ----------------------------
# 4️⃣ Run App
# ----------------------------
if __name__ == "__main__":
    main()








