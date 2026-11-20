"""Generate answers grounded in retrieved handwritten-note passages.

Uses Groq's chat API through LangChain. Configure ``GROQ_API_KEY`` and
optionally ``GENERATOR_MODEL`` in the environment.
"""

import base64
import json
import os
import threading
import time
from collections.abc import Iterator, Sequence
from io import BytesIO
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import SecretStr

from ..Exception import CustomException
from ..logger import Logger
from ..rag.llm_router import LLMRouter


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
            self.api_key = key
            self.model_name = model_name or os.getenv("GENERATOR_MODEL", "openai/gpt-oss-20b")
            # Read at request time too, so a server started before the key was
            # added to .env can recover without reconstructing the pipeline.
            self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
            self.openrouter_model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            self.llm_router = LLMRouter()
            self.llm = ChatGroq(
                model=self.model_name,
                temperature=temperature,
                api_key=SecretStr(key),
            )
            Logger.info("Answer generator initialized with model %s", self.model_name)
        except Exception as error:
            Logger.error("Answer generator initialization failed", exc_info=True)
            raise CustomException(error) from error

    def _openrouter_stream(
        self,
        question: str,
        context: str,
        image: Image.Image,
        system_prompt: str,
        history: Sequence[dict[str, str]] = (),
        cancel_event: threading.Event | None = None,
    ) -> Iterator[str]:
        api_key = os.getenv("OPENROUTER_API_KEY") or self.openrouter_api_key
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for image questions.")

        image_buffer = BytesIO()
        image = image.convert("RGB")
        # The browser sends the original file; cap oversized photos before
        # base64 encoding so image uploads have less latency and payload size.
        image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        image.save(image_buffer, format="JPEG", quality=82)
        prompt = (
            f"Note passages:\n{context or '(No note passages provided.)'}\n\n"
            f"Question:\n{question.strip()}"
        )
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        for turn in history[-6:]:
            role = "assistant" if turn.get("role") == "assistant" else "user"
            content = str(turn.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content})
        image_data = base64.b64encode(image_buffer.getvalue()).decode("ascii")
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
            ],
        })
        body = {
            "model": self.openrouter_model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": True,
            "provider": {"sort": "latency"},
        }
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        encoded_body = json.dumps(body).encode("utf-8")
        response = None
        for attempt, delay in enumerate((0.0, 1.0, 2.0)):
            if delay and cancel_event is not None and cancel_event.wait(delay):
                return
            if delay and cancel_event is None:
                time.sleep(delay)
            request = Request(
                endpoint,
                data=encoded_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "X-Title": "Handwritten Notes RAG",
                },
                method="POST",
            )
            try:
                response = urlopen(request, timeout=120)
                break
            except HTTPError as error:
                if error.code in {429, 500, 502, 503, 504} and attempt < 2:
                    Logger.warning("OpenRouter returned HTTP %s; retrying image generation", error.code)
                    continue
                if error.code in {429, 500, 502, 503, 504}:
                    raise RuntimeError("OpenRouter is temporarily busy. Please try sending the image again in a moment.") from error
                detail = error.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"OpenRouter request failed ({error.code}): {detail}") from error
            except URLError as error:
                if attempt < 2:
                    Logger.warning("OpenRouter connection failed; retrying image generation")
                    continue
                raise RuntimeError(f"Could not reach OpenRouter: {error.reason}") from error

        if response is None:
            raise RuntimeError("OpenRouter could not start the image request. Please try again.")
        try:
            with response:
                for raw_line in response:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    event = json.loads(payload)
                    if event.get("error"):
                        raise RuntimeError(f"OpenRouter stream error: {event['error']}")
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    content = choices[0].get("delta", {}).get("content")
                    if isinstance(content, str) and content:
                        yield content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and isinstance(block.get("text"), str):
                                yield block["text"]
        except URLError as error:
            raise RuntimeError(f"OpenRouter response stream ended unexpectedly: {error.reason}") from error

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
            url = document.get("url")
            if url:
                label += f"; URL: {url}"
            passages.append(f"[{label}]\n{text}")
        return "\n\n".join(passages)

    def generate(
        self,
        question: str,
        documents: Sequence[dict[str, Any]],
        image: Image.Image | None = None,
        grounded: bool = True,
        web_search: bool = False,
        history: Sequence[dict[str, str]] = (),
    ) -> str:
        """Return an evidence-based answer with inline citations and source list.

        If retrieval supplies no usable text, no model call is made and the
        user is told that the notes do not contain evidence for an answer.
        """
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)):
            raise TypeError("documents must be a sequence of retrieved passage mappings")

        context = self._format_context(documents)
        if not context and image is None and grounded and not web_search:
            return "I couldn't find relevant information in your notes to answer that."

        system_prompt = (
            "Answer the user's question using the attached image as primary evidence and the note passages as supporting context. "
            "Carefully read handwriting, mathematical notation, labels, and diagrams. Explain the visible content clearly. "
            "If any handwriting is uncertain, say which part is unclear rather than calling it meaningless or guessing. "
            "Cite note passages only when used; identify the attached image as a source when relevant."
            if image is not None
            else "Answer the user's question using only the provided passages. Do not invent details. "
            "If the passages do not support an answer, say so clearly. Cite supporting passages inline "
            "using their number, such as [1]. End with a Sources list using filenames and pages when available."
            if grounded
            else "You are a helpful assistant. Answer the user's question clearly and accurately."
        )
        if web_search:
            system_prompt += " Use web result passages for current information and cite each relevant result inline as [number]. End with a Sources list including the source title and URL."

        try:
            provider = self.llm_router.select(has_image=image is not None)
            if provider == "openrouter" and image is not None:
                return "".join(self._openrouter_stream(question, context, image, system_prompt, history=history))
            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
            for turn in history[-8:]:
                content = str(turn.get("content", "")).strip()[-1600:]
                if not content:
                    continue
                messages.append(AIMessage(content=content) if turn.get("role") == "assistant" else HumanMessage(content=content))
            messages.extend([
                HumanMessage(content=f"Note passages:\n{context}\n\nQuestion:\n{question.strip()}" if context else question.strip()),
            ])
            model = self.llm.bind(max_tokens=700, reasoning_effort="low", include_reasoning=False) if not grounded else self.llm
            response = model.invoke(messages)
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

    def stream(
        self,
        question: str,
        documents: Sequence[dict[str, Any]],
        image: Image.Image | None = None,
        cancel_event: threading.Event | None = None,
        grounded: bool = True,
        web_search: bool = False,
        history: Sequence[dict[str, str]] = (),
    ) -> Iterator[str]:
        """Yield answer text as the chat model produces it."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        context = self._format_context(documents)
        if not context and image is None and grounded and not web_search:
            yield "I couldn't find relevant information in your notes to answer that."
            return

        system_prompt = (
            "Answer the user's question using the attached image as primary evidence and the note passages as supporting context. "
            "Carefully read handwriting, mathematical notation, labels, and diagrams. Explain the visible content clearly. "
            "If any handwriting is uncertain, say which part is unclear rather than calling it meaningless or guessing. "
            "Cite note passages only when used; identify the attached image as a source when relevant."
            if image is not None
            else "Answer the user's question using only the provided passages. Do not invent details. "
            "If the passages do not support an answer, say so clearly. Cite supporting passages inline "
            "using their number, such as [1]. End with a Sources list using filenames and pages when available."
            if grounded
            else "You are a helpful assistant. Answer the user's question clearly and accurately."
        )
        if web_search:
            system_prompt += " Use web result passages for current information and cite each relevant result inline as [number]. End with a Sources list including the source title and URL."

        try:
            provider = self.llm_router.select(has_image=image is not None)
            if provider == "openrouter" and image is not None:
                yield from self._openrouter_stream(
                    question,
                    context,
                    image,
                    system_prompt,
                    history=history,
                    cancel_event=cancel_event,
                )
                return
            messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
            for turn in history[-8:]:
                content = str(turn.get("content", "")).strip()[-1600:]
                if not content:
                    continue
                messages.append(AIMessage(content=content) if turn.get("role") == "assistant" else HumanMessage(content=content))
            messages.extend([
                HumanMessage(content=f"Note passages:\n{context}\n\nQuestion:\n{question.strip()}" if context else question.strip()),
            ])
            model = (
                self.llm.bind(max_tokens=700, reasoning_effort="low", include_reasoning=False)
                if not grounded
                else self.llm
            )
            if cancel_event is not None:
                try:
                    model = model.bind(stop=["\u0000CANCEL\u0000"])
                except Exception:
                    model = self.llm
            stream = model.stream(messages)
            for chunk in stream:
                if cancel_event is not None and cancel_event.is_set():
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                    return
                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("text"):
                            yield str(block["text"])
        except Exception as error:
            Logger.error("Streaming answer generation failed", exc_info=True)
            raise CustomException(error) from error

    def invoke(self, question: str, documents: Sequence[dict[str, Any]]) -> str:
        """LangChain-style alias for :meth:`generate`."""
        return self.generate(question, documents)
