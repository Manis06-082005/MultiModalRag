"""FastAPI application and first connected chat interface."""

import sys
import json
import base64
import binascii
import asyncio
import queue
import threading
from uuid import uuid4
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
load_dotenv(BASE_DIR / ".env")

from .routes.ingestion import router as ingestion_router
from HandwrittenNotesRAG.logger import Logger

if TYPE_CHECKING:
    from HandwrittenNotesRAG.pipelines.rag_pipeline import RAGPipeline


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_notes: bool = False
    web_search: bool = False
    history: list[ConversationTurn] = Field(default_factory=list, max_length=12)
    conversation_id: str | None = None
    message_id: str | None = None
    image_data: str | None = None
    image_type: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]


@lru_cache(maxsize=1)
def get_rag_pipeline() -> "RAGPipeline":
    """Create the RAG pipeline once, on the first question."""
    from HandwrittenNotesRAG.pipelines.rag_pipeline import RAGPipeline

    return RAGPipeline()


app = FastAPI(
    title="Handwritten Notes Multimodal RAG",
    version="0.1.0",
    description="Ask questions grounded in your ingested notes.",
)
app.include_router(ingestion_router)
_stream_cancellations: dict[str, threading.Event] = {}
_stream_cancellations_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse, tags=["chat"])
def home() -> HTMLResponse:
    """Serve the chat page."""
    return HTMLResponse(
        """
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Noteweave — Your study notes</title>
            <style>
              :root { font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #282923; background: #e9e9e1; }
              * { box-sizing: border-box; }
              body { margin: 0; min-height: 100vh; padding: 24px; }
              header { align-items: center; background: #fff; display: flex; justify-content: space-between; margin: 0 auto; max-width: 1440px; padding: 15px 22px; }
              .brand { align-items: center; display: flex; gap: 11px; }
              .logo { align-items: center; background: #657d68; border-radius: 10px; color: #fff; display: flex; font-family: Georgia,serif; font-size: 18px; height: 36px; justify-content: center; width: 36px; }
              h1 { font-family: Georgia,serif; font-size: 17px; font-weight: 500; margin: 0; }
              header p { color: #77786f; font-size: 11px; margin: 2px 0 0; }
              header a { color: #566e5b; font-size: 12px; text-decoration: none; }
              .layout { background: #fff; border-radius: 22px; display: flex; margin: 0 auto; max-width: 1440px; min-height: calc(100vh - 72px); overflow: hidden; }
              aside { background: #f4f0e6; display: flex; flex: 0 0 290px; flex-direction: column; min-height: 0; overflow: hidden; padding: 24px 16px; }
              .side-title { color: #858277; font-size: 10px; font-weight: 600; letter-spacing: .1em; margin: 25px 0 9px; text-transform: uppercase; }
              .upload-button, .clear-button, #send { border: 0; border-radius: 999px; cursor: pointer; font: inherit; font-weight: 600; padding: 11px 15px; transition: background .15s, transform .15s; }
              .upload-button, #send { background: #657d68; color: white; }
              .upload-button { width: 100%; }
              .upload-button:hover, #send:hover { background: #4d6752; transform: translateY(-1px); }
              .clear-button { background: #e8e4d9; color: #596d5b; width: 100%; }
              .clear-button:hover { background: #ded9ca; }
              .upload-button:disabled, #send:disabled { cursor: wait; opacity: .6; }
              #upload-status, #status { color: #77786f; font-size: 11px; line-height: 1.5; min-height: 18px; }
              #documents { color: #55564f; font-size: 12px; list-style: none; margin: 0; padding: 0; }
              #documents li { border-top: 1px solid #e6e1d5; overflow-wrap: anywhere; padding: 9px 0; }
              #conversation-list { flex: 1; list-style: none; margin: 0 -4px; min-height: 80px; overflow: auto; padding: 0 4px; }
              #conversation-list li { margin: 3px 0; }
              #conversation-list button { background: transparent; border: 0; border-radius: 10px; color: #4e514a; cursor: pointer; display: flex; font: inherit; font-size: 13px; overflow: hidden; padding: 11px 12px; text-align: left; text-overflow: ellipsis; transition: background .16s, transform .16s; white-space: nowrap; width: 100%; }
              #conversation-list button:hover { background: #e9e5da; transform: translateX(2px); }
              #conversation-list button.active { background: #e5e5d9; color: #344b38; font-weight: 600; }
              #clear-history { flex: 0 0 auto; margin-bottom: 14px; }
              .chat { display: flex; flex: 1; flex-direction: column; min-width: 0; padding: clamp(22px, 5vw, 70px); }
              #messages { margin: auto; max-width: 720px; padding: 20px 0; width: 100%; }
              .welcome { margin: 5vh auto 28px; max-width: 580px; text-align: center; }
              .welcome h2 { color: #302b2e; font-family: Georgia,serif; font-size: clamp(28px, 4vw, 42px); font-weight: 500; letter-spacing: -.035em; margin: 0 0 12px; }
              .welcome p { color: #77786f; font-size: 13px; line-height: 1.7; }
              .examples { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 22px; }
              .examples button { background: #fff; border: 1px solid #e8e7df; border-radius: 999px; color: #686a61; cursor: pointer; font: inherit; font-size: 11px; padding: 8px 11px; transition: border .15s,color .15s; }
              .examples button:hover { border-color: #839585; color: #4d6752; }
              .message { animation: message-in .22s ease-out both; display: flex; flex-wrap: wrap; margin: 17px 0; }
              @keyframes message-in { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
              .message.user .bubble { margin-left: auto; }
              .bubble { background: #f4f3ee; border-radius: 15px; line-height: 1.7; max-width: 85%; padding: 13px 16px; white-space: pre-wrap; }
              .user .bubble { background: #657d68; color: white; }
              .user-attachment { border: 1px solid #e1e0d8; border-radius: 14px; display: block; margin: 0 0 8px auto; max-height: 230px; max-width: min(300px, 80%); object-fit: cover; }
              .assistant .bubble { white-space: normal; }
              .assistant .bubble p { margin: 0 0 12px; }
              .assistant .bubble h1, .assistant .bubble h2, .assistant .bubble h3 { color: #33392f; line-height: 1.3; margin: 20px 0 9px; }
              .assistant .bubble h1 { font-size: 1.45em; }
              .assistant .bubble h2 { font-size: 1.25em; }
              .assistant .bubble h3 { font-size: 1.1em; }
              .assistant .bubble ul, .assistant .bubble ol { margin: 7px 0 15px; padding-left: 25px; }
              .assistant .bubble li { margin: 5px 0; padding-left: 3px; }
              .assistant .bubble hr { border: 0; border-top: 1px solid #dfded6; margin: 20px 0; }
              .assistant .bubble pre { background: #292d29; border-radius: 10px; color: #f4f3ed; margin: 12px 0; overflow: auto; padding: 14px; }
              .markdown-table-wrap { margin: 13px 0; max-width: 100%; overflow-x: auto; }
              .markdown-table { border-collapse: collapse; font-size: .94em; min-width: 100%; text-align: left; }
              .markdown-table th, .markdown-table td { border: 1px solid #e1e1d9; padding: 9px 11px; vertical-align: top; }
              .markdown-table th { background: #ebece5; color: #354b3a; font-weight: 650; }
              .markdown-table tr:nth-child(even) td { background: #faf9f6; }
              .message-tools { display: flex; flex-basis: 100%; gap: 7px; margin: 5px 0 0 8px; }
              .message-tools button { background: transparent; border: 0; border-radius: 7px; color: #77786f; cursor: pointer; font: inherit; font-size: 11px; padding: 5px 8px; }
              .message-tools button:hover { background: #f0efe9; color: #354b3a; }
              .assistant .bubble code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .9em; }
              .assistant .bubble :not(pre) > code { background: #e9e9e2; border-radius: 4px; padding: 2px 5px; }
              .assistant .bubble strong { color: #354b3a; }
              .thinking { align-items: center; color: #6e7169; display: inline-flex; gap: 10px; min-height: 24px; }
              .thinking-dots { display: inline-flex; gap: 4px; }
              .thinking-dots i { animation: thinking-pulse 1.15s ease-in-out infinite; background: #788b79; border-radius: 50%; height: 6px; opacity: .35; width: 6px; }
              .thinking-dots i:nth-child(2) { animation-delay: .16s; }
              .thinking-dots i:nth-child(3) { animation-delay: .32s; }
              @keyframes thinking-pulse { 0%, 60%, 100% { opacity: .3; transform: translateY(0); } 30% { opacity: 1; transform: translateY(-3px); } }
              .stream-cursor { animation: cursor-blink 1s steps(2, start) infinite; background: #657d68; border-radius: 2px; display: inline-block; height: 1.15em; margin-left: 3px; transform: translateY(3px); width: 2px; }
              @keyframes cursor-blink { to { visibility: hidden; } }
              .sources { color: #51534b; flex-basis: 100%; font-size: 12px; margin: 8px 0 0 8px; }
              .sources summary { color: #58715d; cursor: pointer; }
              .source-card { background: #f5f4ef; border: 1px solid #ebeae3; border-radius: 9px; margin-top: 8px; padding: 11px; }
              .source-card p { color: #6c6e65; margin: 6px 0 0; white-space: pre-wrap; }
              #chat-form { align-items: end; border: 1px solid #e2e1da; border-radius: 22px; box-shadow: 0 9px 30px #29291b0c; display: flex; gap: 10px; margin: 0 auto; max-width: 760px; padding: 12px; transition: border-color .18s, box-shadow .18s; width: 100%; }
              #chat-form:focus-within { border-color: #a4b19d; box-shadow: 0 10px 34px #29291b14, 0 0 0 3px #657d6814; }
              .web-search-toggle { align-items: center; align-self: center; color: #66695f; cursor: pointer; display: flex; flex: 0 0 auto; font-size: 12px; gap: 5px; white-space: nowrap; }
              .web-search-toggle input { accent-color: #657d68; height: 15px; margin: 0; width: 15px; }
              #attach-button { align-items: center; align-self: end; background: #f1f0eb; border: 0; border-radius: 50%; color: #526b57; cursor: pointer; display: flex; flex: 0 0 40px; font-size: 21px; height: 40px; justify-content: center; transition: background .15s; }
              #attach-button:hover { background: #e5e7dd; }
              #attach-button:disabled { cursor: wait; opacity: .55; }
              #attachment-preview { margin: 0 auto 8px; max-width: 760px; width: 100%; }
              .image-preview-card { align-items: center; animation: preview-in .22s cubic-bezier(.2,.8,.2,1) both; background: #f5f4ef; border: 1px solid #e5e3da; border-radius: 15px; display: flex; gap: 11px; max-width: 280px; padding: 8px; transform-origin: bottom left; }
              @keyframes preview-in { from { opacity: 0; transform: translateY(9px) scale(.96); } to { opacity: 1; transform: translateY(0) scale(1); } }
              .image-preview-card img { background: #e9e8e1; border-radius: 9px; height: 56px; object-fit: cover; width: 64px; }
              .image-preview-meta { flex: 1; min-width: 0; }
              .image-preview-meta strong { color: #363a34; display: block; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
              .image-preview-meta span { color: #7a7b73; display: block; font-size: 11px; margin-top: 4px; }
              .remove-image { align-items: center; background: #e9e7df; border: 0; border-radius: 50%; color: #55584f; cursor: pointer; display: flex; height: 28px; justify-content: center; width: 28px; }
              textarea { border: 0; flex: 1; font: inherit; min-height: 50px; outline: none; padding: 13px 8px; resize: vertical; }
              #send { align-self: end; background: #30252d; border-radius: 50%; font-size: 0; height: 42px; padding: 0; position: relative; width: 42px; }
              #send::after { content: '↑'; font-size: 21px; line-height: 1; }
              #status { margin: 7px auto; max-width: 760px; width: 100%; }
              #menu-button, #drawer-backdrop { display: none; }
              :focus-visible { outline: 3px solid #97ad99; outline-offset: 3px; }
              .layout { width: 100%; }
              aside { min-width: 0; }
              .chat { min-width: 0; }
              #chat-form, textarea { min-width: 0; }
              textarea { overflow-wrap: anywhere; }
              .bubble, .source-card { overflow-wrap: anywhere; }
              @media (max-width: 900px) {
                body { padding: 14px; }
                aside { flex-basis: 230px; padding: 20px 15px; }
                .chat { padding: 28px clamp(18px, 4vw, 42px); }
                .layout { min-height: calc(100vh - 52px); }
              }
              @media (max-width: 620px) {
                body { padding: 0; }
                header { min-height: 58px; padding: 10px 14px; position: sticky; top: 0; z-index: 5; }
                .layout { border-radius: 0; flex-direction: column; min-height: calc(100dvh - 58px); }
                #menu-button { background: transparent; border: 0; color: #526b57; cursor: pointer; display: block; font-size: 22px; margin-right: 9px; padding: 4px; }
                aside { align-items: stretch; border: 0; bottom: 0; box-shadow: 14px 0 40px #1718111a; display: flex; flex: initial; left: 0; max-width: 84vw; overflow: auto; padding: 22px 17px; position: fixed; top: 0; transform: translateX(-105%); transition: transform .24s cubic-bezier(.2,.8,.2,1); width: 300px; z-index: 12; }
                aside.open { transform: translateX(0); }
                #drawer-backdrop { background: #17181166; display: block; inset: 0; opacity: 0; pointer-events: none; position: fixed; transition: opacity .2s; z-index: 11; }
                #drawer-backdrop.open { opacity: 1; pointer-events: auto; }
                .upload-button, .clear-button { flex: 0 0 auto; min-width: 0; padding: 11px 10px; }
                #upload-status, #documents, aside .side-title { display: block; }
                .chat { min-height: calc(100dvh - 116px); padding: 18px 14px 14px; }
                #messages { margin: 0 auto; padding: 12px 0 20px; }
                .welcome { margin: 5vh auto 24px; }
                #chat-form { border-radius: 17px; gap: 7px; padding: 8px; }
                textarea { font-size: 16px; min-height: 44px; padding: 10px 4px; }
                #attach-button { flex-basis: 38px; height: 38px; width: 38px; }
                #send { flex: 0 0 40px; height: 40px; width: 40px; }
                .bubble { max-width: 94%; }
                .sources { margin-left: 0; }
              }
              @media (max-width: 360px) {
                header a { font-size: 11px; }
                .brand { gap: 8px; }
                .upload-button, .clear-button { font-size: 12px; }
                #chat-form { gap: 5px; }
                #attach-button { flex-basis: 34px; height: 34px; width: 34px; }
                #send { flex-basis: 36px; height: 36px; width: 36px; }
              }
              @media (prefers-reduced-motion: reduce) {
                *, *::before, *::after { animation-duration: .01ms !important; scroll-behavior: auto !important; transition-duration: .01ms !important; }
              }
            </style>
          </head>
          <body>
            <header>
              <button id="menu-button" type="button" aria-label="Open chat history" aria-expanded="false">☰</button>
              <div class="brand">
                <div class="logo" aria-hidden="true">N</div>
                <div>
                  <h1>Noteweave</h1>
                  <p>Answers from your study notes</p>
                </div>
              </div>
              <a href="/docs">API docs</a>
            </header>
            <div class="layout">
              <aside>
                <button id="upload-button" class="upload-button" type="button">＋ Attach a file</button>
                <p id="upload-status" role="status">PDFs and images are indexed before you ask about them.</p>
                <button id="clear-history" class="clear-button" type="button">＋ New chat</button>
                <h2 class="side-title">Documents</h2>
                <ul id="documents"></ul>
                <h2 class="side-title">Chat history</h2>
                <ul id="conversation-list"></ul>
              </aside>
              <div id="drawer-backdrop"></div>
              <main class="chat">
                <section id="messages" aria-live="polite"></section>
                <div id="attachment-preview" role="status"></div>
                <form id="chat-form">
                  <input id="pdf-file" type="file" accept=".pdf,.jpg,.jpeg,.png,.bmp,.tiff,.webp,application/pdf,image/*" hidden>
                  <button id="attach-button" type="button" aria-label="Attach PDF or image" title="Attach PDF or image">＋</button>
                  <textarea id="question" maxlength="2000" rows="1"
                    placeholder="Ask a question about your notes..." aria-label="Your question" required></textarea>
                  <label class="web-search-toggle" title="Search the web for current information"><input id="web-search" type="checkbox"><span>Web</span></label>
                  <button id="send" type="submit">Send</button>
                </form>
                <div id="status" role="status"></div>
              </main>
            </div>
            <script>
              const HISTORY_KEY = "notes-ai-chat-history-v1";
              const DOCUMENTS_KEY = "notes-ai-documents-v1";
              const form = document.getElementById("chat-form");
              const questionInput = document.getElementById("question");
              const webSearchToggle = document.getElementById("web-search");
              const sendButton = document.getElementById("send");
              const messages = document.getElementById("messages");
              const status = document.getElementById("status");
              const uploadButton = document.getElementById("upload-button");
              const fileInput = document.getElementById("pdf-file");
              const uploadStatus = document.getElementById("upload-status");
              const documentsList = document.getElementById("documents");
              const attachButton = document.getElementById("attach-button");
              const attachmentPreview = document.getElementById("attachment-preview");
              let activeRequest = null;
              let nearBottom = true;

              function loadSaved(key) {
                try {
                  const value = JSON.parse(localStorage.getItem(key) || "[]");
                  return Array.isArray(value) ? value : [];
                } catch {
                  return [];
                }
              }

              let history = loadSaved(HISTORY_KEY);
              let documents = loadSaved(DOCUMENTS_KEY);
              const CONVERSATIONS_KEY = "notes-ai-conversations-v2";
              let conversations = loadSaved(CONVERSATIONS_KEY);
              let conversationId = localStorage.getItem("notes-ai-current-conversation") || crypto.randomUUID();
              const activeConversation = conversations.find((item) => item.id === conversationId);
              if (activeConversation) history = activeConversation.history || [];
              else if (conversations.length) {
                conversations.sort((a, b) => b.updated - a.updated);
                conversationId = conversations[0].id;
                history = conversations[0].history || [];
              }
              let chatImage = null;

              function saveSaved(key, value) {
                try {
                  localStorage.setItem(key, JSON.stringify(value));
                } catch {
                  status.textContent = "Browser storage is unavailable; history may not persist.";
                }
              }

              function persistConversation() {
                let conversation = conversations.find((item) => item.id === conversationId);
                if (!conversation) {
                  conversation = { id: conversationId, title: history.find((item) => item.role === "user")?.text?.slice(0, 42) || "New conversation", history: [], updated: Date.now() };
                  conversations.unshift(conversation);
                }
                conversation.history = history; conversation.updated = Date.now();
                const firstQuestion = history.find((item) => item.role === "user")?.text;
                if (firstQuestion) conversation.title = firstQuestion.slice(0, 42);
                localStorage.setItem("notes-ai-current-conversation", conversationId);
                saveSaved(CONVERSATIONS_KEY, conversations); saveSaved(HISTORY_KEY, history);
                renderConversationList();
              }

              function renderConversationList() {
                const list = document.getElementById("conversation-list"); list.replaceChildren();
                conversations.slice().sort((a, b) => b.updated - a.updated).forEach((item) => {
                  const row = document.createElement("li"), button = document.createElement("button");
                  button.textContent = item.title || "New conversation";
                  button.classList.toggle("active", item.id === conversationId);
                  button.addEventListener("click", () => {
                    if (activeRequest) { activeRequest.controller.abort(); activeRequest = null; }
                    conversationId = item.id; history = item.history || [];
                    localStorage.setItem("notes-ai-current-conversation", conversationId);
                    renderHistory(); renderConversationList(); closeHistoryDrawer();
                  });
                  row.appendChild(button); list.appendChild(row);
                });
              }

              function closeHistoryDrawer() {
                document.querySelector("aside").classList.remove("open");
                document.getElementById("drawer-backdrop").classList.remove("open");
                document.getElementById("menu-button").setAttribute("aria-expanded", "false");
              }

              function addSources(row, sources) {
                if (!Array.isArray(sources) || sources.length === 0) return;
                const section = document.createElement("section");
                section.className = "sources";
                const details = document.createElement("details");
                const summary = document.createElement("summary");
                summary.textContent = `Sources (${sources.length})`;
                details.appendChild(summary);
                sources.forEach((source, index) => {
                  const card = document.createElement("div");
                  card.className = "source-card";
                  const heading = document.createElement("strong");
                  const page = source.page ? `, page ${source.page}` : "";
                  heading.textContent = `[${index + 1}] ${source.source || "Unknown document"}${page}`;
                  const text = document.createElement("p");
                  text.textContent = source.text || "";
                  card.append(heading, text);
                  if (typeof source.url === "string" && (source.url.startsWith("https://") || source.url.startsWith("http://"))) {
                    const link = document.createElement("a");
                    link.href = source.url;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.textContent = source.url;
                    card.appendChild(link);
                  }
                  details.appendChild(card);
                });
                section.appendChild(details);
                row.appendChild(section);
              }

              function renderMessage(message) {
                const row = document.createElement("article");
                const role = message.role === "user" ? "user" : "assistant";
                row.className = `message ${role}`;
                const bubble = document.createElement("div");
                bubble.className = "bubble";
                if (role === "user" && message.imagePreview) {
                  const image = document.createElement("img");
                  image.className = "user-attachment";
                  image.src = message.imagePreview;
                  image.alt = message.imageName ? `Attached image: ${message.imageName}` : "Attached image";
                  row.appendChild(image);
                }
                if (role === "assistant") renderMarkdown(bubble, message.text || "");
                else bubble.textContent = message.text || "";
                row.appendChild(bubble);
                addSources(row, message.sources);
                if (role === "assistant" && message.text) {
                  const tools = document.createElement("div"); tools.className = "message-tools";
                  const copy = document.createElement("button"); copy.type = "button"; copy.textContent = "Copy answer";
                  copy.addEventListener("click", async () => {
                    try { await navigator.clipboard.writeText(message.text); copy.textContent = "Copied"; }
                    catch { copy.textContent = "Copy unavailable"; }
                    setTimeout(() => { copy.textContent = "Copy answer"; }, 1400);
                  });
                  tools.appendChild(copy); row.appendChild(tools);
                }
                messages.appendChild(row);
              }

              function renderHistory() {
                messages.replaceChildren();
                if (history.length === 0) {
                  const welcome = document.createElement("div");
                  welcome.className = "welcome";
                  const title = document.createElement("h2");
                  title.textContent = "What would you like to learn?";
                  const text = document.createElement("p");
                  text.textContent = "Chat normally, or upload a PDF to ask questions grounded in its content.";
                  welcome.append(title, text);
                  const examples = document.createElement("div");
                  examples.className = "examples";
                  ["How does a recurrence relation work?", "Summarize my notes", "Explain the key idea"].forEach((example) => {
                    const button = document.createElement("button");
                    button.type = "button";
                    button.textContent = example;
                    button.addEventListener("click", () => {
                      questionInput.value = example;
                      form.requestSubmit();
                    });
                    examples.appendChild(button);
                  });
                  welcome.appendChild(examples);
                  messages.appendChild(welcome);
                } else {
                  history.forEach(renderMessage);
                }
                messages.scrollTop = messages.scrollHeight;
              }

              function renderMarkdown(container, markdown) {
                const root = document.createDocumentFragment();
                let list = null, codeLines = null;
                const closeList = () => { if (list) { root.appendChild(list); list = null; } };
                const inline = (parent, text) => {
                  const parts = text.replaceAll("<br/>", "<br>").replaceAll("<br />", "<br>").split("<br>");
                  parts.forEach((part, partIndex) => {
                    if (partIndex) parent.appendChild(document.createElement("br"));
                    const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
                    let cursor = 0, match;
                    while ((match = pattern.exec(part))) {
                      parent.appendChild(document.createTextNode(part.slice(cursor, match.index)));
                      const token = match[0];
                      const tag = token.startsWith("**") ? "strong" : token.startsWith("`") ? "code" : "em";
                      const el = document.createElement(tag);
                      el.textContent = token.slice(tag === "strong" ? 2 : 1, tag === "strong" ? -2 : -1);
                      parent.appendChild(el); cursor = pattern.lastIndex;
                    }
                    parent.appendChild(document.createTextNode(part.slice(cursor)));
                  });
                };
                const lines = markdown.split("\\n");
                const tableCells = (line) => {
                  const cells = line.trim().split("|");
                  if (cells[0].trim() === "") cells.shift();
                  if (cells.length && cells[cells.length - 1].trim() === "") cells.pop();
                  return cells.map((cell) => cell.trim());
                };
                const isTableSeparator = (line) => {
                  const cells = tableCells(line);
                  return cells.length > 1 && cells.every((cell) => cell.length > 0 && [...cell].every((char) => char === "-" || char === ":"));
                };
                for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
                  const line = lines[lineIndex];
                  if (line.trim().startsWith("```")) {
                    closeList();
                    if (codeLines) {
                      const pre = document.createElement("pre"), code = document.createElement("code");
                      code.textContent = codeLines.join("\\n"); pre.appendChild(code); root.appendChild(pre); codeLines = null;
                    } else codeLines = [];
                    continue;
                  }
                  if (codeLines) { codeLines.push(line); continue; }
                  if (line.includes("|") && lineIndex + 1 < lines.length && isTableSeparator(lines[lineIndex + 1])) {
                    closeList();
                    const wrap = document.createElement("div"); wrap.className = "markdown-table-wrap";
                    const table = document.createElement("table"); table.className = "markdown-table";
                    const head = document.createElement("thead"), headRow = document.createElement("tr");
                    tableCells(line).forEach((value) => { const cell = document.createElement("th"); inline(cell, value); headRow.appendChild(cell); });
                    head.appendChild(headRow); table.appendChild(head);
                    const body = document.createElement("tbody");
                    lineIndex += 2;
                    while (lineIndex < lines.length && lines[lineIndex].includes("|") && lines[lineIndex].trim()) {
                      const row = document.createElement("tr");
                      tableCells(lines[lineIndex]).forEach((value) => { const cell = document.createElement("td"); inline(cell, value); row.appendChild(cell); });
                      body.appendChild(row); lineIndex += 1;
                    }
                    lineIndex -= 1;
                    table.appendChild(body); wrap.appendChild(table); root.appendChild(wrap); continue;
                  }
                  const heading = line.match(/^(#{1,3})\s+(.+)$/);
                  if (heading) { closeList(); const el = document.createElement("h" + heading[1].length); inline(el, heading[2]); root.appendChild(el); continue; }
                  if (/^\s*(---+|\*\*\*)\s*$/.test(line)) { closeList(); root.appendChild(document.createElement("hr")); continue; }
                  const bullet = line.match(/^\s*[-*+]\s+(.+)$/);
                  const numbered = line.match(/^\s*\d+[.)]\s+(.+)$/);
                  if (bullet || numbered) {
                    const kind = bullet ? "ul" : "ol";
                    if (list && list.tagName.toLowerCase() !== kind) closeList();
                    if (!list) list = document.createElement(kind);
                    const item = document.createElement("li"); inline(item, (bullet || numbered)[1]); list.appendChild(item); continue;
                  }
                  closeList();
                  if (!line.trim()) continue;
                  const p = document.createElement("p"); inline(p, line); root.appendChild(p);
                }
                closeList();
                if (codeLines) { const pre = document.createElement("pre"), code = document.createElement("code"); code.textContent = codeLines.join("\\n"); pre.appendChild(code); root.appendChild(pre); }
                container.replaceChildren(root);
              }

              function showImagePreview(name, preview) {
                attachmentPreview.replaceChildren();
                const card = document.createElement("div"); card.className = "image-preview-card";
                const image = document.createElement("img"); image.src = preview; image.alt = "Image preview";
                const meta = document.createElement("div"); meta.className = "image-preview-meta";
                const filename = document.createElement("strong"); filename.textContent = name;
                const label = document.createElement("span"); label.textContent = "Ready to send";
                meta.append(filename, label);
                const remove = document.createElement("button"); remove.type = "button"; remove.className = "remove-image";
                remove.setAttribute("aria-label", "Remove attached image"); remove.textContent = "×";
                remove.addEventListener("click", () => { chatImage = null; attachmentPreview.replaceChildren(); fileInput.value = ""; });
                card.append(image, meta, remove); attachmentPreview.appendChild(card);
              }

              function renderDocuments() {
                documentsList.replaceChildren();
                documents.forEach((documentInfo) => {
                  const item = document.createElement("li");
                  item.textContent = documentInfo.filename;
                  documentsList.appendChild(item);
                });
              }

              renderHistory();
              renderDocuments();
              renderConversationList();
              messages.addEventListener("scroll", () => {
                nearBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight < 80;
              });

              document.getElementById("clear-history").addEventListener("click", () => {
                conversationId = crypto.randomUUID(); history = [];
                conversations.unshift({ id: conversationId, title: "New chat", history: [], updated: Date.now() });
                saveSaved(CONVERSATIONS_KEY, conversations);
                renderHistory();
                localStorage.setItem("notes-ai-current-conversation", conversationId);
                renderConversationList(); closeHistoryDrawer(); status.textContent = "New conversation started.";
              });

              document.getElementById("menu-button").addEventListener("click", () => {
                document.querySelector("aside").classList.add("open");
                document.getElementById("drawer-backdrop").classList.add("open");
                document.getElementById("menu-button").setAttribute("aria-expanded", "true");
              });
              document.getElementById("drawer-backdrop").addEventListener("click", closeHistoryDrawer);
              document.addEventListener("keydown", (event) => {
                if (event.key === "Escape") closeHistoryDrawer();
              });

              uploadButton.addEventListener("click", () => fileInput.click());
              attachButton.addEventListener("click", () => {
                fileInput.click();
              });
              fileInput.addEventListener("change", async () => {
                const file = fileInput.files[0];
                if (!file) return;
                if (file.type.startsWith("image/")) {
                  if (file.size > 10 * 1024 * 1024) {
                    attachmentPreview.textContent = "Images must be 10 MB or smaller.";
                    fileInput.value = "";
                    return;
                  }
                  const reader = new FileReader();
                  reader.onload = () => {
                    const dataUrl = String(reader.result);
                    const thumbnail = new Image();
                    thumbnail.onload = () => {
                      const scale = Math.min(1, 480 / thumbnail.width, 320 / thumbnail.height);
                      const canvas = document.createElement("canvas");
                      canvas.width = Math.max(1, Math.round(thumbnail.width * scale));
                      canvas.height = Math.max(1, Math.round(thumbnail.height * scale));
                      canvas.getContext("2d").drawImage(thumbnail, 0, 0, canvas.width, canvas.height);
                      const preview = canvas.toDataURL("image/jpeg", 0.68);
                      chatImage = { data: dataUrl.split(",", 2)[1], type: file.type, name: file.name, preview };
                      showImagePreview(file.name, preview);
                      fileInput.value = "";
                    };
                    thumbnail.onerror = () => { attachmentPreview.textContent = "Could not preview this image."; };
                    thumbnail.src = dataUrl;
                  };
                  reader.onerror = () => { attachmentPreview.textContent = "Could not read this image."; };
                  reader.readAsDataURL(file);
                  return;
                }
                if (file.size > 25 * 1024 * 1024) {
                  uploadStatus.textContent = "Files must be 25 MB or smaller.";
                  fileInput.value = "";
                  return;
                }

                uploadButton.disabled = true;
                attachButton.disabled = true;
                sendButton.disabled = true;
                attachmentPreview.textContent = `Adding ${file.name}…`;
                uploadStatus.textContent = `Uploading ${file.name}...`;
                try {
                  const formData = new FormData();
                  formData.append("file", file);
                  const uploadResponse = await fetch("/api/documents", {
                    method: "POST",
                    body: formData,
                  });
                  const job = await uploadResponse.json();
                  if (!uploadResponse.ok) {
                    throw new Error(job.detail || "Could not upload this file.");
                  }

                  uploadStatus.textContent = "File uploaded. Extracting and indexing text...";
                  while (job.status === "queued" || job.status === "processing") {
                    await new Promise((resolve) => setTimeout(resolve, 2000));
                    const statusResponse = await fetch(`/api/documents/${job.job_id}`);
                    const updatedJob = await statusResponse.json();
                    if (!statusResponse.ok) {
                      throw new Error(updatedJob.detail || "Could not check ingestion status.");
                    }
                    Object.assign(job, updatedJob);
                  }
                  if (job.status !== "completed") {
                    throw new Error(job.error || "File ingestion failed.");
                  }

                  documents = documents.filter((item) => item.filename !== file.name);
                  documents.unshift({ filename: file.name });
                  saveSaved(DOCUMENTS_KEY, documents);
                  renderDocuments();
                  uploadStatus.textContent = `Ready to chat. Indexed ${job.result.stored} chunks.`;
                  attachmentPreview.textContent = `✓ ${file.name} added to your notes`;
                } catch (error) {
                  uploadStatus.textContent = error.message || "PDF upload failed.";
                  attachmentPreview.textContent = "";
                } finally {
                  uploadButton.disabled = false;
                  attachButton.disabled = false;
                  sendButton.disabled = false;
                  fileInput.value = "";
                }
              });

              form.addEventListener("submit", async (event) => {
                event.preventDefault();
                if (activeRequest) {
                  fetch(`/api/query/cancel/${activeRequest.messageId}`, { method: "POST" }).catch(() => {});
                  activeRequest.controller.abort();
                  return;
                }
                const question = questionInput.value.trim();
                if (!question) return;
                const useNotes = documents.some((document) => document.filename?.toLowerCase().endsWith(".pdf"));
                const useWebSearch = webSearchToggle.checked;
                const priorHistory = history
                  .filter((message) => (message.role === "user" || message.role === "assistant") && message.text)
                  .slice(-12)
                  .map((message) => ({ role: message.role, content: message.text.slice(-4000) }));

                history.push({ role: "user", text: question, sources: [], ...(chatImage ? { imagePreview: chatImage.preview, imageName: chatImage.name } : {}) });
                const assistantMessage = { role: "assistant", text: "", sources: [] };
                history.push(assistantMessage);
                renderHistory();
                const assistantBubble = messages.lastElementChild.querySelector(".bubble");
                persistConversation();
                sendButton.disabled = true;
                sendButton.disabled = false;
                sendButton.textContent = "Stop";
                questionInput.disabled = true;
                status.textContent = useWebSearch ? "Searching the web..." : useNotes ? "Searching your notes..." : "Thinking...";

                try {
                  const messageId = crypto.randomUUID();
                  const controller = new AbortController();
                  activeRequest = { messageId, controller };
                  const response = await fetch("/api/query/stream", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    signal: controller.signal,
                    body: JSON.stringify({ question, history: priorHistory, use_notes: useNotes, web_search: useWebSearch, image_data: chatImage?.data || null, image_type: chatImage?.type || null, message_id: messageId, conversation_id: localStorage.getItem("notes-ai-current-conversation") }),
                  });
                  if (!response.ok) {
                    const data = await response.json();
                    throw new Error(typeof data.detail === "string" ? data.detail : "Request failed.");
                  }
                  const reader = response.body.getReader();
                  const decoder = new TextDecoder();
                  let buffer = "";
                  let finished = false;
                  assistantBubble.innerHTML = '<span class="thinking"><span class="thinking-dots" aria-hidden="true"><i></i><i></i><i></i></span><span>Thinking</span></span>';
                  assistantBubble.setAttribute("aria-busy", "true");
                  while (!finished) {
                    const { value, done } = await reader.read();
                    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
                    const events = buffer.split(/\\r?\\n\\r?\\n/);
                    buffer = events.pop() || "";
                    for (const rawEvent of events) {
                      const dataLine = rawEvent.split(/\\r?\\n/).find((line) => line.startsWith("data:"));
                      if (!dataLine) continue;
                      const data = dataLine.slice(5).trimStart();
                      if (data === "[DONE]") { finished = true; break; }
                      const eventData = JSON.parse(data);
                      if (eventData.type === "start") continue;
                      if (eventData.type === "status") { status.textContent = eventData.message; continue; }
                      if (eventData.type === "done" || eventData.type === "cancelled") {
                        finished = true;
                        if (eventData.type === "cancelled") status.textContent = "Generation stopped.";
                        continue;
                      }
                      if (eventData.type === "token") {
                        assistantMessage.text += eventData.content;
                        renderMarkdown(assistantBubble, assistantMessage.text);
                        assistantBubble.setAttribute("aria-busy", "true");
                        const cursor = document.createElement("span");
                        cursor.className = "stream-cursor";
                        cursor.setAttribute("aria-hidden", "true");
                        assistantBubble.appendChild(cursor);
                        if (nearBottom) messages.scrollTop = messages.scrollHeight;
                        status.textContent = "Writing answer…";
                      } else if (eventData.type === "sources") {
                        assistantMessage.sources = eventData.sources || [];
                      } else if (eventData.type === "error") {
                        throw new Error(eventData.message);
                      }
                    }
                    if (done) break;
                  }
                  if (!finished) throw new Error("The stream ended before completion.");
                } catch (error) {
                  if (error.name === "AbortError") status.textContent = "Generation stopped.";
                  else {
                    assistantMessage.text = error.message || "Could not reach the API.";
                    assistantMessage.sources = [];
                    status.textContent = assistantMessage.text;
                  }
                } finally {
                  activeRequest = null;
                  assistantBubble.removeAttribute("aria-busy");
                  if (chatImage) {
                    chatImage = null;
                    attachmentPreview.textContent = "";
                  }
                  renderHistory();
                  persistConversation();
                  sendButton.disabled = false;
                  sendButton.textContent = "Send";
                  questionInput.disabled = false;
                  questionInput.value = "";
                  questionInput.focus();
                  if (["Writing answer…", "Searching your notes...", "Searching the web...", "Thinking..."].includes(status.textContent)) status.textContent = "Ready";
                }
              });

              questionInput.addEventListener("keydown", (event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  form.requestSubmit();
                }
              });
            </script>
          </body>
        </html>
        """
    )


@app.post("/api/query", response_model=QueryResponse, tags=["chat"])
def query_notes(request: QueryRequest) -> QueryResponse:
    """Search the indexed notes and return a grounded answer with sources."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")

    result = get_rag_pipeline().ask(
        question,
        image=_decode_image(request),
        use_notes=request.use_notes,
        web_search=request.web_search,
        history=[turn.model_dump() for turn in request.history],
    )
    return QueryResponse(answer=result["answer"], sources=result["sources"])


@app.post("/api/query/stream", tags=["chat"])
async def stream_query(payload: QueryRequest, request: Request) -> StreamingResponse:
    """Stream answer tokens as server-sent JSON events."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question cannot be blank.")
    image = _decode_image(payload)
    task_id = payload.message_id or uuid4().hex
    cancel_event = threading.Event()
    event_queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
    with _stream_cancellations_lock:
        _stream_cancellations[task_id] = cancel_event

    def run_generation() -> None:
        try:
            event_queue.put({"type": "start", "conversation_id": payload.conversation_id, "message_id": task_id})
            for event in get_rag_pipeline().stream(
                question,
                image=image,
                cancel_event=cancel_event,
                use_notes=payload.use_notes,
                web_search=payload.web_search,
                history=[turn.model_dump() for turn in payload.history],
            ):
                if cancel_event.is_set():
                    break
                event_queue.put(event)
            event_queue.put({"type": "cancelled" if cancel_event.is_set() else "done", "message_id": task_id})
        except Exception as error:
            Logger.error("Streaming chat request failed", exc_info=True)
            cause = getattr(error, "error", None) or error.__cause__ or error
            message = "Answer generation failed. Check the application logs."
            if isinstance(cause, RuntimeError) and (
                str(cause).startswith("OpenRouter is temporarily busy")
                or str(cause).startswith("Could not reach OpenRouter")
                or str(cause).startswith("OpenRouter request failed")
                or str(cause).startswith("OpenRouter stream error")
                or str(cause).startswith("Web search failed")
                or str(cause).startswith("Could not reach Tavily")
            ):
                message = str(cause)
            elif isinstance(cause, ValueError) and str(cause).startswith("OPENROUTER_API_KEY"):
                message = "OpenRouter is not configured. Add OPENROUTER_API_KEY to .env and restart the app."
            elif isinstance(cause, ValueError) and str(cause).startswith("TAVILY_API_KEY"):
                message = "Web search is not configured. Add TAVILY_API_KEY to .env and restart the app."
            event_queue.put({"type": "error", "message": message})
        finally:
            event_queue.put(None)

    async def event_stream():
        worker = threading.Thread(target=run_generation, daemon=True)
        worker.start()
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                try:
                    event = await asyncio.to_thread(event_queue.get, True, 0.2)
                except queue.Empty:
                    continue
                if event is None:
                    break
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            cancel_event.set()
            await asyncio.to_thread(worker.join, 2.0)
            with _stream_cancellations_lock:
                _stream_cancellations.pop(task_id, None)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/query/cancel/{message_id}", tags=["chat"])
def cancel_query(message_id: str) -> dict[str, str]:
    """Cancellation is delivered by aborting the associated SSE connection."""
    with _stream_cancellations_lock:
        cancellation = _stream_cancellations.get(message_id)
    if cancellation is None:
        return {"status": "not_found", "message_id": message_id}
    cancellation.set()
    return {"status": "cancellation_requested", "message_id": message_id}


def _decode_image(request: QueryRequest):
    """Validate and decode an optional browser-provided image payload."""
    if request.image_data is None:
        return None
    if request.image_type not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
        raise HTTPException(status_code=415, detail="Attach a JPEG, PNG, WebP, or GIF image.")
    try:
        raw = base64.b64decode(request.image_data.split(",", 1)[-1], validate=True)
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Images must be 10 MB or smaller.")
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError

        with Image.open(BytesIO(raw)) as opened:
            opened.verify()
        return Image.open(BytesIO(raw)).convert("RGB")
    except HTTPException:
        raise
    except (binascii.Error, ValueError, UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=415, detail="The attached image is invalid.") from error


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Report that the API process is running."""
    return {"status": "ok"}
