"""Real Sentence Transformers embedding tests."""

from HandwrittenNotesRAG.components.embedding import EmbeddingGenerator


def test_embedding_model_creates_normalized_vector() -> None:
    """The real model returns one normalized 384-value vector for a query."""
    model = EmbeddingGenerator()

    vector = model.embed_query("Recurrence relations describe smaller problems.")

    assert len(vector) == 384
    assert all(isinstance(value, float) for value in vector)
    magnitude = sum(value * value for value in vector) ** 0.5
    assert abs(magnitude - 1.0) < 0.001


def test_embedding_model_encodes_multiple_documents() -> None:
    """The real model returns one 384-value vector for each document."""
    model = EmbeddingGenerator()

    vectors = model.embed_document(["First note.", "Second note."])

    assert len(vectors) == 2
    assert all(len(vector) == 384 for vector in vectors)
