"""NotebookLM-style LangChain RAG application for Homework 2.

This app follows the course RAG pattern used in the LangChain lessons:
- load local documents,
- split them into chunks,
- embed chunks into Chroma,
- retrieve relevant chunks for a query,
- answer the user question with a Gemini LLM grounded in the indexed content.

The implementation is intentionally modest and keeps the core retrieval flow while
adding support for a directory of mixed text, Markdown, and PDF documents.
All secrets are expected to come from environment variables rather than being
hard-coded in the source file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class NotebookRAGApp:
    """Basic NotebookLM-like retriever and Q&A application.

    The app stores chunked documents in a Chroma vector database, retrieves the
    most relevant passages for a user question, and then asks a Gemini model to
    answer using that retrieved context. The class also keeps source metadata so
    the returned answer can indicate which document(s) contributed information.
    """

    def __init__(
        self,
        data_dir: str | Path = "./data",
        persist_directory: str | Path = "./.chromadb",
        model_name: str | None = None,
    ) -> None:
        """Initialize the RAG application and all required model and database components."""
        self.data_dir = Path(data_dir)
        self.persist_directory = Path(persist_directory)
        self.model_name = model_name or os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")

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
Use only the context provided below to answer the question.
If the answer is not in the retrieved documents, say that you do not know.
Keep the answer brief, factual, and grounded in the source material.

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
        """Return the value of an environment variable or raise a clear error."""
        value = os.getenv(name)
        if not value:
            raise ValueError(f"Missing required environment variable: {name}")
        return value

    @staticmethod
    def _format_docs(docs: Sequence[Document]) -> str:
        """Combine retrieved document pages into a single prompt context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _ensure_source_metadata(self, documents: Iterable[Document]) -> list[Document]:
        """Ensure each document has a source metadata field so provenance is preserved."""
        normalized_docs: list[Document] = []
        for doc in documents:
            metadata = dict(doc.metadata or {})
            source = metadata.get("source") or metadata.get("file_path") or "unknown"
            metadata["source"] = source
            normalized_docs.append(Document(page_content=doc.page_content, metadata=metadata))
        return normalized_docs

    def load_documents(self, directory: str | Path | None = None) -> list[Document]:
        """Load supported documents from a directory and return standardized LangChain documents.

        Supported file types include .txt, .md, and .pdf. A directory may contain a
        mix of those formats, and all documents will be loaded into the same RAG
        knowledge base for a single search index.
        """
        target_dir = Path(directory) if directory is not None else self.data_dir

        if not target_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {target_dir}")

        txt_loader = DirectoryLoader(str(target_dir), glob="**/*.txt", loader_cls=TextLoader)
        md_loader = DirectoryLoader(str(target_dir), glob="**/*.md", loader_cls=TextLoader)
        pdf_loader = DirectoryLoader(str(target_dir), glob="**/*.pdf", loader_cls=PyPDFLoader)

        documents: list[Document] = []
        for loader in (txt_loader, md_loader, pdf_loader):
            documents.extend(self._ensure_source_metadata(loader.load()))

        if not documents:
            raise ValueError(
                f"No supported documents were found in {target_dir}. "
                "Supported extensions: .txt, .md, .pdf"
            )

        return documents

    def index_documents(self, documents: Iterable[Document]) -> None:
        """Chunk documents and add them to the persistent Chroma vector database."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(list(documents))

        if not chunks:
            raise ValueError("No document chunks were created for indexing.")

        self.vectorstore.add_documents(chunks)

    def list_sources(self) -> list[str]:
        """Return the unique source document names currently indexed in the vector store."""
        collection_data = self.vectorstore.get()
        sources = {meta.get("source", "unknown") for meta in collection_data.get("metadatas", [])}
        return sorted(sources)

    def _get_sources_for_question(self, question: str) -> list[str]:
        """Return source names for the top matching chunks for a query."""
        matches = self.vectorstore.similarity_search(question, k=4)
        sources = []
        for doc in matches:
            source = (doc.metadata or {}).get("source")
            if source and source not in sources:
                sources.append(source)
        return sources

    def answer_question(self, question: str) -> str:
        """Ask the RAG chain a question and append the source documents it used."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        answer = self.rag_chain.invoke(question)
        sources = self._get_sources_for_question(question)

        if sources:
            return f"{answer}\n\nSources: {', '.join(sources)}"
        return answer

    def run(self) -> None:
        """Run the application in a simple CLI mode.

        The user can provide a directory containing text, Markdown, and PDF files.
        The app will index the files into one Chroma collection and then answer
        questions grounded in the retrieved document chunks.
        """
        print(f"Loading documents from: {self.data_dir}")
        documents = self.load_documents(self.data_dir)
        print(f"Loaded {len(documents)} source document(s).")

        print("Indexing documents into the Chroma vector store...")
        self.index_documents(documents)

        print("Indexed sources:")
        for source in self.list_sources():
            print(f"  - {source}")

        print("\nAsk a question about the indexed documents. Type 'exit' or 'quit' to end.")
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
    """Command-line entry point for the RAG application."""
    parser = argparse.ArgumentParser(description="NotebookLM-like LangChain RAG application")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing .txt, .md, and .pdf documents to index.",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./.chromadb",
        help="Directory used to persist the Chroma vector store.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional Gemini model name. Overrides GOOGLE_MODEL and defaults to gemini-3.6-flash.",
    )
    args = parser.parse_args()

    app = NotebookRAGApp(
        data_dir=args.data_dir,
        persist_directory=args.persist_dir,
        model_name=args.model_name,
    )
    app.run()


if __name__ == "__main__":
    main()
