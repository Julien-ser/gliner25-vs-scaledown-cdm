#!/usr/bin/env python3
"""Run the exact CDM example inputs through GLiNER2.5-small locally.

Same tickets and invoice as ~/workspace/cdm_example.py, so results can be
compared directly to ScaleDown's published Classify/Extract numbers.
No API keys, no network after the model download.
"""
import json, time, sys

MODEL_ID = "fastino/gliner2.5-small-v1"

TICKETS = [
    {"id": "t1", "text": "I was charged twice for my subscription this month"},
    {"id": "t2", "text": "the export button does nothing when I click it"},
    {"id": "t3", "text": "password reset email never arrives"},
    {"id": "t4", "text": "any good pasta recipes?"},
]
LABELS = ["billing", "bug", "account", "other"]

INVOICE = """INVOICE #INV-20817
Bill to: Fenwick & Daughters Ltd
Items: Industrial pallet racking x 12 @ $412.00; Delivery x 1 @ $240.00
Total due $5,351.88. Payment terms: Net 30. Remit to: Lakeshore Fabrication Co."""

def main():
    t0 = time.time()
    print(f"Loading {MODEL_ID} ...", flush=True)
    from gliner2 import AutoExtractor
    model = AutoExtractor.from_pretrained(MODEL_ID, map_location="cpu")
    print(f"Loaded {type(model).__name__} in {time.time()-t0:.1f}s\n", flush=True)

    results = {"model": MODEL_ID, "classify": [], "extract": None}

    # --- Classify: use the constrained Classifier for probabilities ---
    print("== 1. Classify (GLiNER2.5 Classifier, probabilities) ==", flush=True)
    try:
        from gliner2.classification import Classifier, ClassificationSchema
        clf = Classifier.from_pretrained(MODEL_ID, map_location="cpu")
        schema = ClassificationSchema().single("issue_type", LABELS)
        for t in TICKETS:
            s = time.time()
            r = clf.classify(t["text"], schema)
            d = r.to_dict()
            print(json.dumps(d, indent=2))
            issue = d.get("issue_type", {})
            probs = issue.get("probabilities", {})
            top = issue.get("value")
            conf = issue.get("confidence")
            action = "route" if (conf is not None and conf >= 0.8) else "-> REVIEW (score < 0.8 or no score)"
            print(f"  {t['id']}  {top}  conf={conf}  {action}  ({time.time()-s:.2f}s)")
            results["classify"].append({"id": t["id"], "text": t["text"], "top": top, "confidence": conf, "probabilities": probs, "feasible": d.get("_meta", {}).get("feasible")})
    except Exception as e:
        print(f"Classifier failed: {type(e).__name__}: {e}", flush=True)
        # fallback to classify_text
        print("Falling back to classify_text (label only, no probabilities)", flush=True)
        for t in TICKETS:
            r = model.classify_text(t["text"], {"issue_type": LABELS})
            print(f"  {t['id']} -> {r}")
            results["classify"].append({"id": t["id"], "text": t["text"], "raw": str(r)})

    # --- Also try plain classify_text for comparison ---
    print("\n== 1b. classify_text (simple API) ==", flush=True)
    for t in TICKETS:
        try:
            r = model.classify_text(t["text"], {"issue_type": LABELS})
            print(f"  {t['id']} -> {r}")
        except Exception as e:
            print(f"  {t['id']} failed: {e}")

    # --- Extract invoice ---
    print("\n== 2. Extract (GLiNER2.5 extract_entities) ==", flush=True)
    entities = {
        "vendor": "Company payment is due to / remit to",
        "total_due": "Total amount due",
        "po_number": "Customer purchase order number",
        "bill_to": "Company being billed",
        "invoice_number": "Invoice number",
    }
    s = time.time()
    res = model.extract_entities(INVOICE, entities, include_confidence=True, include_spans=True)
    print(json.dumps(res, indent=2, default=str))
    print(f"({time.time()-s:.2f}s)")
    results["extract"] = res
    # Check spans are real
    ents = res.get("entities", {}) if isinstance(res, dict) else {}
    for field in ["vendor", "total_due", "po_number"]:
        vals = ents.get(field) or []
        if vals:
            for v in vals:
                ok = INVOICE[v["start"]:v["end"]] == v["text"] if "start" in v else "?"
                print(f"  {field} -> {v.get('text')!r} conf={v.get('confidence')} span_ok={ok}")
        else:
            print(f"  {field} -> None / not found (nothing invented)")

    with open("gliner_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nTotal elapsed: {time.time()-t0:.1f}s. Saved gliner_results.json")

if __name__ == "__main__":
    main()
