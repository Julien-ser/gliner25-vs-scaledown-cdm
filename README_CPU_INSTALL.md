# GLiNER2.5 — CPU-only install (avoid the CUDA download that failed here)

Why this order matters: in this session, `pip install "gliner2[local]"`
tried to download the 539 MB CUDA PyTorch wheel and the connection dropped
at 257 MB, so nothing ran and no model was downloaded. `gliner2[local]`
pulls in PyTorch, so installing CPU-only PyTorch *first* means pip sees
torch as already satisfied and does not fetch the CUDA wheel.

Tested against: Python 3.10+ required (gliner2 docs); model
`fastino/gliner2.5-small-v1` (74M, English, fast CPU).

## On your own computer

```bash
python3 --version        # needs 3.10+
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 1. CPU-only PyTorch FIRST — official PyTorch CPU index
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Then GLiNER2 local (reuses the torch you just installed)
pip install "gliner2[local]"

# 3. Quick check
python3 -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
# expect cuda: False on a CPU-only install

# 4. Run the ready test (same 4 tickets + invoice INV-20817 as the ScaleDown example)
python3 run_gliner.py
```

If a download drops again, just re-run the same command — pip will resume/
retry from where it can; avoid starting a fresh CUDA install in parallel.

## What to compare against (ScaleDown's published numbers)

Classify — 0.8 cutoff, below it goes to human review:
| ticket | expected top label | score |
|---|---|---|
| t1 "charged twice…" | billing | 0.993 |
| t2 "export button does nothing…" | bug | 0.991 |
| t3 "password reset email never arrives" | account | 0.562 → REVIEW |
| t4 "any good pasta recipes?" | other | 0.982 |

Extract — invoice INV-20817, Fenwick & Daughters Ltd, $5,351.88 due:
- vendor → "Lakeshore Fabrication Co." (0.996, span in the input)
- total_due → "$5,351.88" (0.787)
- po_number → None (no PO number on the invoice — the key check: it must
  come back empty, not invented, unlike the LLM JSON-schema run that
  returned "null" as a string, then "INV-20817")

What will probably differ: ScaleDown's own benchmark reports +15.5 points
on classification vs GLiNER2.5 but only +0.2 points on extraction, so
expect the extraction numbers to be close and the classification scores —
especially t3 — to be the interesting divergence. Send back the output
(or `gliner_results.json`, which the script writes) and Rio can compare
it line-by-line against these numbers.
