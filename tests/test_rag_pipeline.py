from HandwrittenNotesRAG.pipelines.rag_pipeline import RAGPipeline


def test_stream_handles_extended_hi_without_initializing_generator() -> None:
    pipeline = RAGPipeline()

    events = list(pipeline.stream("hii", use_notes=False))

    assert events == [
        {"type": "token", "content": "Hi! How can I help you today?"},
        {"type": "sources", "sources": []},
    ]
