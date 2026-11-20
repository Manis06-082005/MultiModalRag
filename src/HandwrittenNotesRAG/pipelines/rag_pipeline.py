"""Retrieve note passages and generate a source-grounded answer."""

from collections.abc import Iterator
import threading
from typing import Any
from PIL import Image

from ..components.generator import Generator
from ..components.retriever import Retriever
from ..components.web_search import TavilyWebSearch
from .retrieval_pipeline import RetrievalPipeline


class RAGPipeline:
    """Run hybrid retrieval followed by Groq answer generation."""

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline | None = None,
        generator: Generator | None = None,
        retriever: Retriever | None = None,
        web_searcher: TavilyWebSearch | None = None,
    ) -> None:
        if retrieval_pipeline is not None and retriever is not None:
            raise ValueError("Pass retrieval_pipeline or retriever, not both")
        self.retrieval_pipeline = retrieval_pipeline
        self.retriever = retriever
        self.web_searcher = web_searcher or TavilyWebSearch()
        self.generator = generator or Generator()

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        if self.retrieval_pipeline is not None:
            return self.retrieval_pipeline.retrieve(question)
        if self.retriever is not None:
            return self.retriever.search(question)
        self.retrieval_pipeline = RetrievalPipeline()
        return self.retrieval_pipeline.retrieve(question)

    def ask(
        self,
        question: str,
        image: Image.Image | None = None,
        use_notes: bool = True,
        web_search: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Return the generated answer alongside the passages used as sources."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        passages = self._retrieve(question.strip()) if use_notes else []
        if web_search:
            passages.extend(self.web_searcher.search(question.strip()))
        if image is not None:
            passages = [*passages, {"text": "Image supplied directly to the vision model.", "source": "Attached image"}]
        answer = self.generator.generate(
            question.strip(),
            passages,
            image=image,
            grounded=use_notes or image is not None or web_search,
            web_search=web_search,
            history=history or [],
        )
        return {"answer": answer, "sources": passages}

    def stream(
        self,
        question: str,
        image: Image.Image | None = None,
        cancel_event: threading.Event | None = None,
        use_notes: bool = True,
        web_search: bool = False,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield answer tokens followed by the retrieved source passages."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        clean_question = question.strip()
        if web_search:
            yield {"type": "status", "message": "Searching the web..."}
        passages = self._retrieve(clean_question) if use_notes else []
        if web_search:
            passages.extend(self.web_searcher.search(clean_question))
        if image is not None:
            passages = [*passages, {"text": "Image supplied directly to the vision model.", "source": "Attached image"}]
        for token in self.generator.stream(
            clean_question,
            passages,
            image=image,
            cancel_event=cancel_event,
            grounded=use_notes or image is not None or web_search,
            web_search=web_search,
            history=history or [],
        ):
            if cancel_event is not None and cancel_event.is_set():
                yield {"type": "cancelled"}
                return
            yield {"type": "token", "content": token}
        if cancel_event is not None and cancel_event.is_set():
            yield {"type": "cancelled"}
            return
        yield {"type": "sources", "sources": passages}

    def run(self, question: str) -> dict[str, Any]:
        """Alias for :meth:`ask`."""
        return self.ask(question)
