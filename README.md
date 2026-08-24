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
rag-triage  9 cases  |  16 chunks  |  judge lexical  |  k=5

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

The `support rank` column is the position the supporting chunk gets in a plain BM25
ranking of the question. For a `retrieval_miss` that number is the actionable part: if
it says `#9` and you are retrieving `k=5`, raising k fixes the case; if it says `-`, no
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
chunks, the supporting one highlighted, and the sentence breakdown.

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
- Per-sentence grounding: which sentence of the answer no retrieved chunk supports
- Works from recorded traces: put the chunk ids your production retriever returned in
  the eval set and rag-triage grades those instead of running its own search
- `--explain <case-id>` for the full trace of a single case
- Text, markdown, HTML and JSON reports; `--strict` exits non-zero for CI
- Call, token and cost accounting for the Claude judge

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

Every run with the Claude judge ends with a `judge usage` line: how many API calls it
made, how many of those were served from the in-process cache, the input and output
tokens, and an estimated cost from the published per-token price for the model. The
bundled nine-case example needs 89 judge calls, or 80 with `--no-claim-detail`.

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
  retriever.py     BM25 index and tokenizer
  judge.py         Judge protocol and the offline lexical judge
  claude_judge.py  Claude judge, same protocol, plus usage accounting
  claims.py        sentence splitting and per-sentence grounding
  triage.py        the decision tree that assigns a verdict
  report.py        text, markdown, HTML and JSON rendering
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
- Sentence splitting is a regex with an abbreviation list. It handles decimals, version
  numbers and `e.g.`, and it will merge two sentences when the second starts with a
  lowercase word. Bulleted or tabular answers are treated as one sentence.
- Judge cost is reported per run, not per case, and the Claude judge does not use prompt
  caching yet even though the system prompt is identical on every call.
