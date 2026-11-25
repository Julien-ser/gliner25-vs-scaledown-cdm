# GLiNER2.5 vs ScaleDown CDM — free local model, same test

Can a free, local, open-weight model do the job of a paid Constrained
Decoding Model (CDM) API? This project runs the exact inputs from
ScaleDown's CDM example — 4 support tickets and invoice #INV-20817 —
through **GLiNER2.5-small** (Fastino, 74M params, Apache 2.0) running
locally on CPU, and compares the output line-by-line against ScaleDown's
published Classify / Extract numbers.

No API keys. No paid service. No network needed after the model download.

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/Julien-ser/gliner25-vs-scaledown-cdm.git
cd gliner25-vs-scaledown-cdm
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 run_gliner.py            # writes gliner_results.json
```

`requirements.txt` pulls CPU-only PyTorch from the PyTorch CPU index.
That matters: a plain `pip install "gliner2[local]"` drags in the
539 MB CUDA PyTorch wheel even on machines without a GPU.

## Results — actual run, 2026-11-24 (CPU)

Python 3.14.7 · torch 2.15.0+cpu · gliner2 2.2.0 ·
`fastino/gliner2.5-small-v1` · ~23 s total.
Raw output from this run: [`gliner_results.json`](gliner_results.json).

### Classify — 0.8 cutoff, below it goes to human review

| ticket | ScaleDown | GLiNER2.5-small | verdict |
|---|---|---|---|
| t1 "charged twice…" | billing 0.993 | billing **0.994** | match |
| t2 "export button does nothing…" | bug 0.991 | bug **0.967** | match |
| t3 "password reset email never arrives" | account 0.562 → REVIEW | bug **0.826** | **mismatch — confidently wrong** |
| t4 "any good pasta recipes?" | other 0.982 | other **0.457** | same label, much lower confidence |

### Extract — invoice #INV-20817

| field | ScaleDown | GLiNER2.5-small |
|---|---|---|
| total_due | $5,351.88 (0.787) | $5,351.88 (**0.9995**) ✓ |
| invoice_number | — | correct ✓ |
| po_number | None | correctly empty ✓ — no hallucination |
| vendor / bill_to | vendor Lakeshore Fabrication Co. (0.996) ✓ | **swapped at high confidence** — Fenwick as vendor, Lakeshore as bill-to, exactly backwards |

For contrast, ScaleDown's own LLM-with-JSON-schema run invented a
`po_number` (`"null"` as a string, then the invoice number `INV-20817`).
GLiNER passed that no-hallucination test; the strict-JSON LLM did not.

## Takeaway

On clear cases the free local model is basically the paid CDM — same
labels, near-identical confidence, and it nails the invoice total.
Its failures aren't hesitant failures, they're high-confidence ones:
t3 would auto-route to the wrong team instead of going to review, and
the vendor/bill-to swap would pay the wrong company. A confidence
cutoff won't save you there — that's exactly where a CDM's calibrated
scores (t3 at 0.562 → review) earn their keep.

## Files

- `run_gliner.py` — the test: same tickets + invoice as ScaleDown's example
- `requirements.txt` — pinned deps, CPU-only PyTorch index
- `gliner_results.json` — raw output of the 2026-11-24 run above
  (the script overwrites it on each run)

## License

MIT — see [LICENSE](LICENSE).
