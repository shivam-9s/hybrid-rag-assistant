from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
import numpy as np
import os
import pickle


def load_hybrid_retriever():

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load FAISS
    vectorstore = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.docstore._dict
    documents = [docs[k] for k in docs]

    texts = [doc.page_content for doc in documents]

    # Build BM25
    tokenized_corpus = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    def hybrid_search(query, k=4):

        # Vector search
        vector_results = vectorstore.similarity_search(query, k=k)

        # BM25 search
        tokenized_query = query.split()
        bm25_scores = bm25.get_scores(tokenized_query)

        top_bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        bm25_results = [documents[i] for i in top_bm25_indices]

        # Merge results (remove duplicates)
        combined = list({doc.page_content: doc for doc in vector_results + bm25_results}.values())

        return combined[:k]

    return hybrid_search