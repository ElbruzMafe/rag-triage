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
rag-triage  8 cases  |  16 chunks  |  judge lexical  |  k=5

  case               verdict              support rank
  -----------------  -------------------  ------------
  scim-provisioning  missing_from_corpus  -
  webhook-retry      retrieval_miss       #1
  refund-window      generation_miss      #1
  backup-retention   ungrounded           #1
  downgrade-timing   ungrounded           #1
  seat-counting      no_answer            #1
  rate-limit         ok                   #1
  page-size          ok                   #1

summary
  missing_from_corpus 1
  retrieval_miss      1
  generation_miss     1
  ungrounded          2
  no_answer           1
  ok                  2

what to fix first
  2x ungrounded -> the answer states things no retrieved chunk supports - tighten the prompt or add a citation check
  1x missing_from_corpus -> the evidence is not in the corpus - fix ingestion, not the prompt
  1x retrieval_miss -> evidence exists but never reached the model - fix chunking, embeddings or k
  1x generation_miss -> the model had the evidence and still got it wrong - fix the prompt or the model
  1x no_answer -> no answer recorded for this case - run the system under test first
```

The `support rank` column is the position the supporting chunk gets in a plain BM25
ranking of the question. For a `retrieval_miss` that number is the actionable part: if
it says `#9` and you are retrieving `k=5`, raising k fixes the case; if it says `-`, no
amount of k helps and the chunking or the embedding is the problem.

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
- Two judges behind one interface: an offline token-overlap judge, and a Claude judge
  for eval sets whose wording differs from the corpus
- Evidence location - finds which chunks back the gold answer, anywhere in the corpus
- Works from recorded traces: put the chunk ids your production retriever returned in
  the eval set and rag-triage grades those instead of running its own search
- Text, markdown and JSON reports; `--strict` exits non-zero for CI

## Tech stack

Python 3.10+, PyYAML, and the `anthropic` SDK for the optional Claude judge. Tests use
pytest. There is no vector database and no embedding model - see DECISIONS.md.

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

Reports:

```bash
rag-triage --corpus docs/ --evalset eval.yaml --json out.json --markdown report.md --strict
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
  retriever.py     BM25 index and tokenizer
  judge.py         Judge protocol and the offline lexical judge
  claude_judge.py  Claude judge, same protocol
  triage.py        the decision tree that assigns a verdict
  report.py        text, markdown and JSON rendering
  cli.py           argparse entry point
examples/          sample docs and an eval set covering every verdict
tests/             pytest suite
```

## Known gaps

- The lexical judge is token overlap plus a hard check on numbers. It cannot tell a
  paraphrase from a contradiction, so a wrong answer that reuses the corpus wording can
  be reported as `ok`. The Claude judge exists for this; the lexical one is the default
  because it needs no key and runs in milliseconds.
- The stemmer is a handful of suffix rules, not Porter. It gets plurals and `-ed`/`-ing`
  right and will mangle irregular words.
- Retrieval is BM25 only, so `retrieval_miss` verdicts describe a lexical baseline, not
  your embedding model. Feeding recorded `retrieved_ids` from your own pipeline is the
  accurate path today; a pluggable retriever backend is the next piece of work.
- Evidence location scans the top `--support-scan` BM25 candidates rather than the whole
  corpus, so a gold answer sharing no vocabulary with its source chunk can be reported
  as `missing_from_corpus`.
- No per-case cost accounting for the Claude judge yet.
