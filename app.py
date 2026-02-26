import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from rag_chain import build_rag_chain


st.set_page_config(page_title="Hybrid RAG Assistant", layout="wide")

st.title("📄 Hybrid RAG Document Assistant")
st.write("Upload a PDF and ask questions about it.")

# session states
if "processed" not in st.session_state:
    st.session_state.processed = False

if "messages" not in st.session_state:
    st.session_state.messages = []


# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(docs, embeddings)

            vectorstore.save_local("vector_store")

        st.session_state.processed = True
        st.success("Document processed successfully!")


# 🚫 Block questions before processing
if not st.session_state.processed:
    st.warning("⚠ Please upload and process a PDF before asking questions.")
else:

    qa_chain = build_rag_chain()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question about the document...")

    if query:

        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Thinking..."):
            answer = qa_chain.invoke(query)

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})