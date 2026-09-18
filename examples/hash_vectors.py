#!/usr/bin/env python3
"""Turn a --embed-inputs file into a --vectors file, without an embedding model.

This is a hashing vectorizer: every word is hashed to a dimension and counted. It has
no semantic understanding at all - "refund" and "reimbursement" land in unrelated
dimensions - so it is here to exercise the --vectors plumbing, not to be a good
retriever. Replace the body of `embed()` with a call to whatever model you actually
serve, keep the keys identical, and the numbers in the report become about that model.

    rag-triage --corpus examples/corpus --evalset examples/evalset.yaml \
        --embed-inputs inputs.json
    python examples/hash_vectors.py inputs.json vectors.json
    rag-triage --corpus examples/corpus --evalset examples/evalset.yaml \
        --vectors vectors.json
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys

DIM = 256
WORD = re.compile(r"[a-z0-9]+")


def embed(text: str) -> list[float]:
    vector = [0.0] * DIM
    for word in WORD.findall(text.lower()):
        digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % DIM
        sign = 1.0 if digest[4] % 2 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        # rag-triage rejects zero vectors, and an empty text is a real input error
        raise ValueError(f"nothing to embed in {text!r}")
    return [round(v / norm, 6) for v in vector]


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2

    with open(argv[1], encoding="utf-8") as handle:
        inputs = json.load(handle)

    payload = {
        "model": "hash-demo",
        "chunks": {cid: embed(text) for cid, text in inputs["chunks"].items()},
        "queries": {text: embed(text) for text in inputs["queries"]},
    }
    with open(argv[2], "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    print(
        f"embedded {len(payload['chunks'])} chunks and {len(payload['queries'])} queries "
        f"into {DIM} dimensions -> {argv[2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
