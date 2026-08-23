# Decisions

## Root cause instead of a score

Existing eval tools (Ragas, DeepEval, promptfoo) report metrics: faithfulness,
answer relevancy, context precision. A number tells you something is wrong but not
which of the three stages to open, and the three stages need unrelated fixes. I chose a
single categorical verdict per case instead, because "the evidence is not in your
corpus" is an instruction and "faithfulness 0.62" is not. The cost of that choice is
that rag-triage says less about *how* wrong an answer is.

## The oracle is the gold answer, not the question

To decide whether retrieval failed, you have to know where the correct evidence lives.
Retrieving on the question cannot tell you that - if the question retrieves nothing
useful, that is the very failure being diagnosed. So evidence location searches with the
*gold answer* as the query, which shares vocabulary with the source passage by
construction, and then judges the candidates. It is the one part of the design that
makes the whole decision tree possible.

## BM25 rather than embeddings

An embedding retriever means torch or an API key, a model download, and a vector store,
for a tool whose job is diagnosis rather than serving. BM25 in ~60 lines has no install
story at all and makes the whole test suite run in under a second. The honest cost:
`retrieval_miss` describes a lexical baseline, not the user's actual retriever. Two
things mitigate it - an eval case can carry the `retrieved_ids` its real pipeline
returned, in which case the verdict is about that pipeline, and the retriever sits
behind a narrow interface (`search`, `score_all`) so a vector backend is an addition,
not a rewrite.

## A tiny stemmer, written by hand

Without stemming, "refunds" in a document does not match "refund" in a question, and the
tool reports retrieval failures that are really word endings. NLTK's Porter stemmer
would fix that and add a dependency plus a corpus download. Six suffix rules cover
plurals, `-ed` and `-ing`, which is where the mismatches actually were, and they are
short enough to read and reason about. Irregular forms stay broken; that is written
down in the README rather than hidden.

## Two judges behind one protocol

Every semantic question the tool asks - is this claim supported, do these two answers
agree, is this answer grounded - goes through a three-method `Judge` protocol. The
default `LexicalJudge` is token overlap and is deterministic, free and offline, so the
test suite and CI never touch the network. `ClaudeJudge` implements the same protocol
against Claude Opus 5 for eval sets whose wording is far from the corpus. Keeping the
protocol at three methods was deliberate: it is the smallest surface that supports the
decision tree, and it means adding a judge is one class.

## A hard check on numbers

Token overlap treats "5 business days" and "14 business days" as near identical, and a
swapped figure is the single most common RAG failure I found while reading incident
write-ups. So the lexical judge extracts numbers from both strings and forces a
mismatch when they disagree, regardless of overlap. It is a special case rather than a
general solution, and it catches the failure that matters most.

## Structured outputs for the Claude judge

The judge needs a boolean, a confidence and a reason. Parsing that out of prose is
fragile, so the request pins a JSON schema via `output_config.format` and the response
is guaranteed to parse. Judge calls are also memoised per process, because several eval
cases sharing a gold answer ask the same question about the same chunk.

## Evidence search is capped

Judging every chunk against every gold answer is O(chunks x cases) judge calls, which
is fine for the lexical judge and ruinous for the Claude one. BM25 on the gold answer is
used as a recall filter and only the top `--support-scan` candidates are judged. This is
a real precision/cost trade: a gold answer that shares no vocabulary with its source
chunk will be reported as `missing_from_corpus`. The flag is exposed so the trade is the
user's to make.

## Recorded traces as a first-class input

The most useful mode is not "let rag-triage retrieve for you", it is "here is what my
production retriever returned, grade it". That is why `retrieved_ids` exists on an eval
case. When it is present the retrieval verdict is about the real system, and the BM25
ranking is reported only as a baseline for comparison - which is itself informative: if
BM25 puts the evidence at #1 and the production retriever missed it, the embedding
setup is worth a look.

## argparse and no framework

One command with ten flags does not need click or typer. The whole CLI is one file and
`main(argv)` takes its arguments as a list, so the tests call it directly instead of
spawning subprocesses.
