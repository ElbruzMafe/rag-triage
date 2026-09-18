"""Ranking by cosine similarity over vectors the user supplies from their own model."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .models import Chunk, EvalCase, Hit


def load_vectors(
    path: str | Path,
) -> tuple[dict[str, list[float]], dict[str, list[float]], str | None]:
    """Read a vectors file: chunk vectors, query vectors, and the model that made them.

    The format is deliberately dumb - two mappings of key to list of floats - so any
    embedding model can produce it with a few lines of script.
    """
    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"vectors file does not exist: {file}")
    raw = file.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not valid JSON ({exc})") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object with 'chunks' and 'queries'")

    chunks = _vector_map(data.get("chunks"), "chunks", path)
    queries = _vector_map(data.get("queries") or {}, "queries", path)
    if not chunks:
        raise ValueError(f"{path}: 'chunks' is empty - there is nothing to search")

    _check_dimensions({**chunks, **queries}, path)

    model = data.get("model")
    return chunks, queries, str(model) if model else None


class VectorRetriever:
    """Cosine similarity over precomputed vectors, with the BM25 retriever's interface.

    rag-triage never calls an embedding model itself; that would mean an API key or a
    model download for a tool whose job is diagnosis. Instead the vectors come in from
    whatever model the user actually runs in production, which is the point: a
    `retrieval_miss` from this retriever is about their pipeline, not a lexical proxy.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        chunk_vectors: dict[str, list[float]],
        query_vectors: dict[str, list[float]],
        model: str | None = None,
    ):
        self.chunks = list(chunks)
        self.name = f"vectors:{model}" if model else "vectors"
        self._queries = query_vectors

        missing = [chunk.id for chunk in self.chunks if chunk.id not in chunk_vectors]
        if missing:
            sample = ", ".join(missing[:3])
            raise ValueError(
                f"{len(missing)} of {len(self.chunks)} chunks have no vector (e.g. {sample}). "
                "The vectors file was probably built from a different chunking - "
                "run --embed-inputs with the same --max-chars and --overlap."
            )

        self._vectors = []
        self._norms = []
        self.dim = len(chunk_vectors[self.chunks[0].id]) if self.chunks else 0
        for chunk in self.chunks:
            vector = chunk_vectors[chunk.id]
            if len(vector) != self.dim:
                raise ValueError(
                    f"vector for chunk {chunk.id!r} has {len(vector)} dimensions, "
                    f"the others have {self.dim}"
                )
            norm = math.sqrt(sum(value * value for value in vector))
            if norm == 0.0:
                raise ValueError(f"vector for chunk {chunk.id!r} is all zeros")
            self._vectors.append(vector)
            self._norms.append(norm)

    @property
    def size(self) -> int:
        return len(self.chunks)

    def score_all(self, query: str) -> list[Hit]:
        """Every chunk whose cosine similarity is positive, best first."""
        vector = self._queries.get(query)
        if vector is None:
            raise LookupError(
                f"no vector for the text {_clip(query)!r}. Every question and gold answer "
                "in the eval set needs one - use --embed-inputs to list the texts to embed."
            )
        if len(vector) != self.dim:
            raise ValueError(
                f"query vector for {_clip(query)!r} has {len(vector)} dimensions, "
                f"the chunk vectors have {self.dim}"
            )
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise ValueError(f"query vector for {_clip(query)!r} is all zeros")

        scored = []
        for i, chunk in enumerate(self.chunks):
            similarity = _dot(vector, self._vectors[i]) / (norm * self._norms[i])
            if similarity > 0:
                scored.append((similarity, chunk))

        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [
            Hit(chunk=chunk, score=round(score, 4), rank=rank)
            for rank, (score, chunk) in enumerate(scored, start=1)
        ]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return self.score_all(query)[:k]


def embed_inputs(chunks: list[Chunk], cases: list[EvalCase]) -> dict:
    """The texts a vectors file has to cover: every chunk, every question, every gold.

    Gold answers are in there because evidence location searches with the gold answer
    rather than the question - see DECISIONS.md.
    """
    queries: list[str] = []
    seen: set[str] = set()
    for case in cases:
        for text in (case.question, case.gold):
            if text not in seen:
                seen.add(text)
                queries.append(text)
    return {
        "chunks": {chunk.id: chunk.searchable for chunk in chunks},
        "queries": queries,
    }


def _vector_map(value, field: str, path) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: '{field}' must be an object of key -> vector")

    vectors = {}
    for key, vector in value.items():
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError(f"{path}: '{field}' entry {key!r} is not a non-empty list of numbers")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vector):
            raise ValueError(f"{path}: '{field}' entry {key!r} contains a non-numeric value")
        vectors[str(key)] = [float(v) for v in vector]
    return vectors


def _check_dimensions(vectors: dict[str, list[float]], path) -> None:
    sizes: dict[int, str] = {}
    for key, vector in vectors.items():
        sizes.setdefault(len(vector), key)
        if len(sizes) > 1:
            (dim_a, key_a), (dim_b, key_b) = sorted(sizes.items())[:2]
            raise ValueError(
                f"{path}: vectors have different dimensions - "
                f"{key_a!r} has {dim_a}, {key_b!r} has {dim_b}"
            )


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _clip(text: str, limit: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
