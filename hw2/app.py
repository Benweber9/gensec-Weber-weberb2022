"""Minimal NotebookLM-style RAG application.

This script follows the structure used in the course's LangChain RAG examples:
- load documents from a local directory,
- split them into chunks,
- embed and persist them in Chroma,
- retrieve relevant chunks for a user question,
- answer the question with a Google Gemini model.

The implementation is intentionally small and focused on the core RAG workflow.
It does not yet include advanced features like chat history, multi-document UI,
file uploads, or agent orchestration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class NotebookRAGApp:
    """Basic NotebookLM-like retriever and question-answering application.

    The app mirrors the course RAG architecture by loading local text documents,
    splitting them into smaller chunks, embedding those chunks with Gemini
    embeddings, saving them in a persistent Chroma collection, and using a
    retriever to ground answers in the indexed content.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        persist_directory: str | Path = "./.chromadb",
        model_name: str | None = None,
    ) -> None:
        """Initialize the app and configure the database, LLM, and retrieval chain."""
        self.data_dir = Path(data_dir)
        self.persist_directory = Path(persist_directory)
        self.model_name = model_name or os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

        self.google_project = self._require_env("GOOGLE_CLOUD_PROJECT")
        self.google_api_key = self._require_env("GOOGLE_API_KEY")

        self.embedding_function = VertexAIEmbeddings(
            model_name="gemini-embedding-001",
            project=self.google_project,
            location="us-west1",
        )

        self.vectorstore = Chroma(
            persist_directory=str(self.persist_directory),
            embedding_function=self.embedding_function,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.google_api_key,
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful research assistant.
Use only the provided context to answer the question.
If the answer is not in the context, say that you do not know.
Keep the response brief and grounded in the retrieved material.

Question: {question}

Context: {context}

Answer:"""
        )

        self.rag_chain = (
            {
                "context": self.retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    @staticmethod
    def _require_env(name: str) -> str:
        """Return an environment variable value or raise a clear error if missing."""
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    @staticmethod
    def _format_docs(docs: Sequence[Document]) -> str:
        """Convert retrieved documents into a single string for the prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    def load_documents(self) -> list[Document]:
        """Load the documents from the configured directory and return them as LangChain documents."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        txt_loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        md_loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
        )

        documents: list[Document] = []
        for loader in (txt_loader, md_loader):
            documents.extend(loader.load())

        if not documents:
            raise ValueError(f"No supported documents were found in {self.data_dir}")

        return documents

    def index_documents(self, documents: Iterable[Document]) -> None:
        """Split documents into chunks and add them to the persistent Chroma vector store."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(list(documents))

        if not chunks:
            raise ValueError("No document chunks were created for indexing.")

        self.vectorstore.add_documents(chunks)

    def list_sources(self) -> list[str]:
        """Return the unique source names currently in the vector database."""
        collection_data = self.vectorstore.get()
        sources = {meta.get("source", "unknown") for meta in collection_data.get("metadatas", [])}
        return sorted(sources)

    def answer_question(self, question: str) -> str:
        """Use the retriever and LLM to answer a question grounded in the indexed documents."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        return self.rag_chain.invoke(question)

    def run(self) -> None:
        """Run a simple interactive CLI workflow for document indexing and retrieval."""
        print("Loading documents from:", self.data_dir)
        documents = self.load_documents()
        print(f"Loaded {len(documents)} source document(s).")

        print("Indexing documents into the Chroma vector store...")
        self.index_documents(documents)

        print("Indexed sources:")
        for source in self.list_sources():
            print(f"  - {source}")

        print("\nAsk a question about the indexed documents. Type 'exit' to quit.")
        while True:
            question = input("Question> ").strip()
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            answer = self.answer_question(question)
            print("\nAnswer:\n")
            print(answer)
            print()


def main() -> None:
    """Entry point for the minimal RAG application."""
    app = NotebookRAGApp(
        data_dir="./data",
        persist_directory="./.chromadb",
    )
    app.run()


if __name__ == "__main__":
    main()
