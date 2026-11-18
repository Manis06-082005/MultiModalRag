"""Generate answers grounded in retrieved handwritten-note passages.

Uses Groq's chat API through LangChain. Configure ``GROQ_API_KEY`` and
optionally ``GENERATOR_MODEL`` in the environment.
"""

import os
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from ..Exception import CustomException
from ..logger import Logger


class Generator:
    """Answer questions using retrieved note chunks as the only evidence."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        try:
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("GROQ_API_KEY is required to initialize Generator")
            self.model_name = model_name or os.getenv("GENERATOR_MODEL", "openai/gpt-oss-20b")
            self.llm = ChatGroq(
                model=self.model_name,
                temperature=temperature,
                api_key=key,
            )
            Logger.info("Answer generator initialized with model %s", self.model_name)
        except Exception as error:
            Logger.error("Answer generator initialization failed", exc_info=True)
            raise CustomException(error) from error

    @staticmethod
    def _format_context(documents: Sequence[dict[str, Any]]) -> str:
        passages = []
        for index, document in enumerate(documents, start=1):
            text = str(document.get("text", "")).strip()
            if not text:
                continue
            source = document.get("source") or document.get("source_file") or document.get("file_name") or document.get("filename") or "Unknown source"
            page = document.get("page") or document.get("page_number")
            label = f"Passage {index}; source: {source}"
            if page is not None:
                label += f"; page: {page}"
            passages.append(f"[{label}]\n{text}")
        return "\n\n".join(passages)

    def generate(self, question: str, documents: Sequence[dict[str, Any]]) -> str:
        """Return an evidence-based answer with inline citations and source list.

        If retrieval supplies no usable text, no model call is made and the
        user is told that the notes do not contain evidence for an answer.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of retrieved passage mappings")

        context = self._format_context(documents)
        if not context:
            return "I couldn't find relevant information in your notes to answer that."

        messages = [
            SystemMessage(
                content=(
                    "Answer the user's question using only the provided note passages. "
                    "Do not invent details. If the passages do not support an answer, "
                    "say so clearly. Cite supporting passages inline using their "
                    "number, such as [1]. End with a 'Sources' list mapping each "
                    "citation used to the source filename and page when available. "
                    "Never invent a source or page. Keep the answer direct and readable."
                )
            ),
            HumanMessage(content=f"Note passages:\n{context}\n\nQuestion:\n{question.strip()}"),
        ]
        try:
            response = self.llm.invoke(messages)
            answer = response.content
            if isinstance(answer, str):
                return answer.strip()
            # Chat models may return a list of text blocks.
            return "".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in answer
            ).strip()
        except Exception as error:
            Logger.error("Answer generation failed", exc_info=True)
            raise CustomException(error) from error

    def invoke(self, question: str, documents: Sequence[dict[str, Any]]) -> str:
        """LangChain-style alias for :meth:`generate`."""
        return self.generate(question, documents)
