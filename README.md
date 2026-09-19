# rag-triage

Takes an eval set and the answers your RAG system gave, and tells you *why* each wrong
answer was wrong: the evidence was never in the corpus, it was in the corpus but never
retrieved, or it was retrieved and the model still got it wrong.

## Why I built it

Every RAG debugging session I have seen starts the same way: someone pastes a bad
answer into Slack and the team argues about whether it is a chunking problem, an
embedding problem or a prompt problem. Existing eval tools score the answer - they give
you a faithfulness number between 0 and 1 - but a score does not tell you which part of
the pipeline to open. The three failures need three completely different fixes, and
guessing wrong costs a day.

So rag-triage does not score answers. It answers one question per case: which stage
broke. It finds where the evidence for the gold answer actually lives in your corpus,
checks whether that chunk reached the model, and only then looks at what the model did
with it.

## Example output

Run against the sample docs in `examples/`:

```
rag-triage  9 cases  |  16 chunks  |  retriever bm25  |  judge lexical  |  k=5

  case               verdict              support rank
  -----------------  -------------------  ------------
  scim-provisioning  missing_from_corpus  -
  webhook-retry      retrieval_miss       #1
  refund-window      generation_miss      #1
  backup-retention   ungrounded           #1
  downgrade-timing   ungrounded           #1
  incident-sla       ungrounded           #1
  seat-counting      no_answer            #1
  page-size          ok                   #1
  rate-limit         ok                   #1

summary
  missing_from_corpus 1
  retrieval_miss      1
  generation_miss     1
  ungrounded          3
  no_answer           1
  ok                  2

what to fix first
  3x ungrounded -> the answer states things no retrieved chunk supports - tighten the prompt or add a citation check
  1x missing_from_corpus -> the evidence is not in the corpus - fix ingestion, not the prompt
  1x retrieval_miss -> evidence exists but never reached the model - fix chunking, embeddings or k
  1x generation_miss -> the model had the evidence and still got it wrong - fix the prompt or the model
  1x no_answer -> no answer recorded for this case - run the system under test first
```

The `support rank` column is the position the supporting chunk gets when the retriever
ranks the question. For a `retrieval_miss` that number is the actionable part: if it
says `#9` and you are retrieving `k=5`, raising k fixes the case; if it says `-`, no
amount of k helps and the chunking or the embedding is the problem.

## Drilling into one case

The table tells you which stage broke; `--explain` tells you why it says so.

```
$ rag-triage --corpus examples/corpus --evalset examples/evalset.yaml --explain incident-sla

case     incident-sla
verdict  ungrounded  (stage: generation)
fix      the answer states things no retrieved chunk supports - tighten the prompt or add a citation check

question
  How quickly are incidents published after they are detected?
gold
  Incidents are published on the status page within 15 minutes of detection.
answer
  Incidents are published on the status page within 60 minutes of detection. Our on-call engineer phones every affected customer before anything is posted.

evidence for the gold answer
  #1  operations.md#2     score 20.283    Status and incidents

retrieved context (3 chunks)
  #1  operations.md#2     score 6.7672    Status and incidents  <- supporting
  #2  api.md#3            score 2.7242    Webhooks
  #3  api.md#1            score 1.6598    Rate limits

answer sentences
  ok  [operations.md#2] Incidents are published on the status page within 60 minutes of detection.
  !!  [unsupported] Our on-call engineer phones every affected customer before anything is posted.

notes
  - answer mismatch: figures disagree: gold has ['15'], answer has ['60']
  - groundedness: 8/17 claim terms present in passage (across 3 retrieved passages)
  - unsupported sentence: 'Our on-call engineer phones every affected customer before anything is posted.'
```

The `answer sentences` block is the part worth having. "This answer is ungrounded" is
a label; "this sentence is the invented one, and here is the chunk that backs the other
one" is a fix. Note that the first sentence is marked supported even though its figure
is wrong - grounded and correct are different questions, and the figure is caught one
line above by the answer mismatch. Per-sentence checks only run when the whole-answer
groundedness check has already failed, so they cost nothing on healthy cases.

`--html report.html` writes the same detail for every case as one self-contained file:
counts per verdict at the top, then a collapsible card per case with the retrieved
chunks, the supporting one highlighted, and the sentence breakdown. The verdict tiles
are also filters - clicking one hides the other cases. No JavaScript, no external
requests, so it survives being attached to a ticket or a CI run.

## The verdicts

| verdict | what happened | where to look |
| --- | --- | --- |
| `ok` | the answer matches the gold answer | - |
| `missing_from_corpus` | nothing in the corpus supports the gold answer | ingestion, not the prompt |
| `retrieval_miss` | the evidence exists but was not in the retrieved context | chunking, embeddings, k |
| `generation_miss` | the evidence was in context and the answer is still wrong | prompt or model |
| `ungrounded` | the answer states things no retrieved chunk supports | prompt, citation checks |
| `no_answer` | the eval set has no recorded answer for this case | run your system first |

An `ok` case whose answer is not supported by the retrieved context gets a note: the
model probably knew the answer already, which means the case is not testing retrieval
at all.

## Features

- Heading-aware markdown chunking, with paragraph packing and a hard-split fallback
- Dependency-free BM25 retriever with a small English stemmer
- A vector backend behind the same interface: feed it your own embeddings and the
  verdicts describe your retriever, not a lexical stand-in
- Two judges behind one interface: an offline token-overlap judge, and a Claude judge
  for eval sets whose wording differs from the corpus
- Evidence location - finds which chunks back the gold answer, anywhere in the corpus
- Per-sentence grounding: which sentence of the answer no retrieved chunk supports
- Works from recorded traces: put the chunk ids your production retriever returned in
  the eval set and rag-triage grades those instead of running its own search
- `--explain <case-id>` for the full trace of a single case
- `--compare` puts two retrievers side by side on the same cases, and names the ones
  a retriever change fixes, breaks, or cannot help
- Text, markdown, HTML and JSON reports; `--strict` exits non-zero for CI
- Call, token and cost accounting for the Claude judge

## Tech stack

Python 3.10+, PyYAML, and the `anthropic` SDK for the optional Claude judge. Tests use
pytest. There is no vector database, and rag-triage calls no embedding model itself - it
consumes vectors you supply. See DECISIONS.md.

## Running it

```bash
git clone https://github.com/ElbruzMafe/rag-triage
cd rag-triage
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

rag-triage --corpus examples/corpus --evalset examples/evalset.yaml -k 5
pytest
```

With the Claude judge:

```bash
pip install -e ".[claude]"
export ANTHROPIC_API_KEY=...        # see .env.example
rag-triage --corpus examples/corpus --evalset examples/evalset.yaml --judge claude
```

Every run with the Claude judge ends with a `judge usage` line: how many API calls it
made, how many of those were served from the in-process cache, the input and output
tokens, and an estimated cost from the published per-token price for the model. The
bundled nine-case example needs 89 judge calls, or 80 with `--no-claim-detail`.

### Grading your own retriever

By default the `retrieval_miss` verdicts describe BM25, which is probably not what you
serve. To make them describe your embedding model, hand rag-triage the vectors. It never
calls an embedding model itself - it only consumes what you give it.

First ask what needs embedding. The answer depends on the chunking, so pass the same
`--max-chars` and `--overlap` you will run with:

```bash
rag-triage --corpus docs/ --evalset eval.yaml --embed-inputs inputs.json
# wrote 16 chunk texts and 18 query texts to inputs.json
```

`inputs.json` is `{"chunks": {id: text}, "queries": [text, ...]}`. Run your model over
it and write the vectors back in the same shape:

```json
{
  "model": "text-embedding-3-small",
  "chunks":  {"api.md#0": [0.01, -0.02, ...]},
  "queries": {"What is the API rate limit?": [0.03, ...]}
}
```

```bash
rag-triage --corpus docs/ --evalset eval.yaml --vectors vectors.json
# rag-triage  9 cases  |  16 chunks  |  retriever vectors:text-embedding-3-small  |  ...
```

Queries are both the questions and the gold answers, because evidence location searches
with the gold answer rather than the question. Chunk ids encode the chunking, so a
vectors file built with different chunk settings is rejected with a count of what is
missing instead of quietly ranking a subset.

`examples/hash_vectors.py` is a stdlib hashing vectorizer for trying the flow without an
API key. It has no semantic understanding and is not a serious retriever; it exists so
you can see the wiring before writing the real embedder:

```bash
python examples/hash_vectors.py inputs.json vectors.json
```

### Comparing two retrievers

Deciding whether a new embedding model is worth shipping means knowing which cases it
fixes and which it breaks. `--compare` runs the same eval set through the BM25 baseline
and your vectors, and prints the two verdicts side by side:

```bash
rag-triage --corpus docs/ --evalset eval.yaml --vectors vectors.json --compare
```

```
rag-triage comparison  9 cases  |  16 chunks  |  judge lexical  |  k=5
  A  bm25
  B  vectors:hash-demo

  case               A verdict            B verdict            A rank  B rank
  -----------------  -------------------  -------------------  ------  ------
  backup-retention   ungrounded           ungrounded           #1      #1
  page-size          ok                   ok                   #1      #1
  rate-limit         ok                   ok                   #1      #1
  refund-window      generation_miss      generation_miss      #1      #1
  scim-provisioning  missing_from_corpus  missing_from_corpus  -       -
  webhook-retry      retrieval_miss       retrieval_miss       #1      #6

changed
  0 of 9 cases get a different verdict

retrieval
  A got the evidence to the model in 7 of 9 cases, B in 7
  biggest rank moves: webhook-retry #1 -> #6
```

That is the hashing vectorizer, so the null result is the correct one: it has no
semantic understanding, it is not meaningfully different from lexical matching on this
corpus, and no verdict moves. What it does show is the rank column. On `webhook-retry`
the supporting chunk sits at #1 for BM25 and #6 for the vectors, so that case is one k
away from breaking even though today both retrievers call it the same thing. Rank moves
without a verdict change are the headroom a case has left.

A case marked `*` is one where the verdict actually changed. `--strict` exits 1 when B
loses evidence that A got into the context, which makes the comparison usable as a CI
gate on a retriever change. `--json` writes the same data; `--markdown` and `--html` are
single-run reports and are rejected here.

Evidence is located once per case, with A, and handed to both sides. Whether the gold
answer is backed by the corpus at all is a fact about the corpus, so it must not change
just because the retriever under test changed - otherwise a weak embedding reports an
ingestion bug instead of its own recall problem. It also halves the judge calls.

Reports:

```bash
rag-triage --corpus docs/ --evalset eval.yaml \
  --json out.json --markdown report.md --html report.html --strict
```

### Eval set format

```yaml
cases:
  - id: refund-window
    question: How long does a refund take to arrive?
    gold: Refunds are returned to the original payment card within 5 business days.
    answer: Refunds are returned to the original payment card within 14 business days.
    retrieved_ids: ["billing.md#2"]   # optional, from your production trace
    tags: [billing]                   # optional
```

`answer` is what your system produced. Leave it out and the case still reports whether
the evidence exists and whether it is retrievable.

## Project structure

```
ragtriage/
  models.py        dataclasses shared by every stage
  corpus.py        document loading, chunking, eval set parsing
  retriever.py     Retriever protocol, BM25 index and tokenizer
  vectors.py       cosine retriever over vectors you supply
  judge.py         Judge protocol and the offline lexical judge
  claude_judge.py  Claude judge, same protocol, plus usage accounting
  claims.py        sentence splitting and per-sentence grounding
  triage.py        the decision tree that assigns a verdict
  compare.py       two retrievers over one eval set, paired by case
  report.py        text, markdown, HTML and JSON rendering
  cli.py           argparse entry point
examples/          sample docs, an eval set covering every verdict, hash_vectors.py
tests/             pytest suite
```

## Known gaps

- The lexical judge is token overlap plus a hard check on numbers. It cannot tell a
  paraphrase from a contradiction, so a wrong answer that reuses the corpus wording can
  be reported as `ok`. The Claude judge exists for this; the lexical one is the default
  because it needs no key and runs in milliseconds.
- The stemmer is a handful of suffix rules, not Porter. It gets plurals and `-ed`/`-ing`
  right and will mangle irregular words.
- The default retriever is BM25, so unless you pass `--vectors` or recorded
  `retrieved_ids`, a `retrieval_miss` describes a lexical baseline rather than your
  embedding model. The vector backend closes that, but it makes you produce the vectors
  yourself - there is no `--embed-with` that calls a provider for you.
- The vector retriever holds every vector in memory and scores them one at a time. That
  is fine for the thousands of chunks an eval corpus has and wrong for a real index;
  there is no ANN structure behind it.
- Query vectors are looked up by exact text, so an eval set edited after the vectors were
  built fails loudly on the changed case rather than re-embedding it.
- Evidence location scans the top `--support-scan` candidates the retriever returns for
  the gold answer, rather than the whole corpus, so a gold answer sharing no vocabulary with its source chunk can be reported
  as `missing_from_corpus`.
- Sentence splitting is a regex with an abbreviation list. It handles decimals, version
  numbers and `e.g.`, and it will merge two sentences when the second starts with a
  lowercase word. Bulleted or tabular answers are treated as one sentence.
- Judge cost is reported per run, not per case. Prompt caching is deliberately absent:
  the system prompt plus a chunk is around 300 tokens, well under the ~1024-token minimum
  cacheable prefix, so `cache_control` would have cost a line of code and saved nothing.
  DECISIONS.md has the measurement.
