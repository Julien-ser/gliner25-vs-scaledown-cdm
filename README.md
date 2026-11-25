# GLiNER2.5 vs ScaleDown CDM — free local model, same test

Run the exact Constrained Decoding Model (CDM) example inputs — 4 support
tickets and invoice #INV-20817 — through GLiNER2.5-small locally, no API
keys, no paid service, and compare line-by-line against ScaleDown's
published Classify / Extract numbers.

## Run it

CPU-only PyTorch **first** — `pip install "gliner2[local]"` on its own
pulls the 539 MB CUDA PyTorch wheel, which failed here at 257 MB.
Full install steps: see `README_CPU_INSTALL.md`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "gliner2[local]"
python3 run_gliner.py   # writes gliner_results.json
```

Model: `fastino/gliner2.5-small-v1` (74M, English, fast on CPU).
No network needed after the model download.

## Results — run 2026-11-24 on julienlab (CPU)

Python 3.14.7, torch 2.15.0+cpu, gliner2 2.2.0, ~23 s total.
Full raw output: Vault `Muse/raw/2026-11-24_gliner25-results.md`.

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
- `README_CPU_INSTALL.md` — CPU-first install guide + ScaleDown numbers to compare against
- `gliner_results.json` — written by the script on each run (not committed)
