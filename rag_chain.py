from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from retriever import load_hybrid_retriever
import os

def build_rag_chain():

    # Load retriever
    hybrid_retriever = load_hybrid_retriever()

    # Groq Llama3 model
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY")
    )

    template = """
You are an intelligent AI assistant.

Use the provided context to answer the question clearly.

If the answer exists in the context:
- Explain it in your own words.
- Do NOT copy raw text.
- Provide a clear explanation.

If the context does not contain the answer, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": lambda x: format_docs(hybrid_retriever(x)),
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain