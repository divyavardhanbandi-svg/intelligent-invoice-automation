"""
Intelligent Invoice Automation
==============================
End-to-end pipeline:
    read invoice (txt / pdf / image) -> extract fields -> validate -> categorize (ML)
    -> detect duplicates & anomalies -> route (auto-approve / review / reject) -> export

Install
-------
    pip install pandas numpy scikit-learn
    # optional, only for real PDFs / scanned images:
    pip install pdfplumber pytesseract pillow      (images also need the Tesseract binary)

Usage
-----
    # Option 1: generate sample invoices and process them (no setup needed)
    python invoice_automation.py

    # Option 2: process your own folder of invoices (.txt, .pdf, .png, .jpg)
    python invoice_automation.py --input my_invoices/

Outputs (in ./outputs)
----------------------
    processed_invoices.csv   every invoice with extracted fields, category, status, reasons
    review_queue.csv         only the invoices a human needs to look at
"""

import argparse
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

OUTPUT_DIR = Path("outputs")
SAMPLE_DIR = Path("sample_invoices")

AUTO_APPROVE_LIMIT = 10000.0      # invoices above this always go to a human
MIN_CATEGORY_CONFIDENCE = 0.40    # below this, the predicted category is not trusted
AMOUNT_TOLERANCE = 0.02           # allowed rounding difference in totals


# --------------------------------------------------------------------------
# 1. Sample data (stands in for real vendor invoices)
# --------------------------------------------------------------------------
CATALOG = {
    "Office Supplies": ["Printer paper A4", "Ballpoint pens box", "Stapler and staples",
                        "Desk organizer", "Whiteboard markers", "Sticky notes pack",
                        "Ink cartridge black"],
    "IT & Software": ["Cloud hosting subscription", "Software license renewal",
                      "Laptop docking station", "Network switch 24 port",
                      "Antivirus annual license", "Domain registration renewal",
                      "Monitor 27 inch"],
    "Travel": ["Flight ticket economy", "Hotel accommodation per night",
               "Taxi airport transfer", "Car rental weekly", "Train ticket business",
               "Conference travel meal allowance"],
    "Utilities": ["Electricity usage monthly", "Water supply charges",
                  "Internet broadband monthly", "Natural gas usage",
                  "Waste collection service", "Phone line charges"],
    "Marketing": ["Social media advertising campaign", "Brand design services",
                  "Print flyers brochure", "Trade show booth rental",
                  "Email marketing platform", "Search engine ad spend"],
    "Professional Services": ["Legal consulting hours", "Accounting audit services",
                              "Management consulting retainer", "Recruitment agency fee",
                              "Training workshop facilitation", "Tax advisory services"],
}
VENDORS = {
    "Acme Office Supplies": "Office Supplies",
    "BrightByte Technologies": "IT & Software",
    "SkyWays Travel Co": "Travel",
    "CityPower and Water": "Utilities",
    "Spark Media Agency": "Marketing",
    "Sterling Advisory LLP": "Professional Services",
}
PRICE_RANGE = {
    "Office Supplies": (3, 40), "IT & Software": (30, 400), "Travel": (50, 350),
    "Utilities": (60, 250), "Marketing": (100, 500), "Professional Services": (150, 600),
}
# Three different invoice layouts so the parser is not tuned to a single template
STYLES = [
    dict(no="Invoice No", date="Invoice Date", fmt="%Y-%m-%d", sub="Subtotal",
         tax="Tax", total="Total Due", header="INVOICE"),
    dict(no="Invoice #", date="Date", fmt="%b %d, %Y", sub="Sub Total",
         tax="VAT", total="Amount Due", header="Tax Invoice"),
    dict(no="Invoice Number", date="Invoice Date", fmt="%d %B %Y", sub="Subtotal",
         tax="GST", total="Grand Total", header="INVOICE"),
]


def finalize(inv):
    """Recompute amounts from the line items so a generated invoice is consistent."""
    inv["lines"] = [(d, q, u, round(q * u, 2)) for d, q, u, _ in inv["lines"]]
    inv["subtotal"] = round(sum(l[3] for l in inv["lines"]), 2)
    inv["tax"] = round(inv["subtotal"] * inv["rate"], 2)
    inv["total"] = round(inv["subtotal"] + inv["tax"], 2)
    return inv


def render_invoice(inv):
    s = STYLES[inv["style"]]
    due = inv["date"] + timedelta(days=30)
    out = [inv["vendor"], "123 Market Street, Springfield", "", s["header"]]
    if inv["inv_no"]:
        out.append(f"{s['no']}: {inv['inv_no']}")
    out.append(f"{s['date']}: {inv['date']:{s['fmt']}}")
    out.append(f"Due Date: {due:{s['fmt']}}")
    if inv["po"]:
        out.append(f"PO Number: {inv['po']}")
    out += ["", "Description                    Qty   Unit Price      Amount"]
    for desc, qty, unit, amt in inv["lines"]:
        out.append(f"{desc:<30} {qty:>3} ${unit:>10,.2f} ${amt:>10,.2f}")
    out += [
        "",
        f"{s['sub']}: ${inv['subtotal']:,.2f}",
        f"{s['tax']} ({inv['rate']:.0%}): ${inv['tax']:,.2f}",
        f"{s['total']}: ${inv['total']:,.2f}",
        "",
        "Thank you for your business.",
    ]
    return "\n".join(out)


def generate_sample_invoices(folder, n=30, seed=7):
    """Create n text invoices, with a few deliberate problems for the pipeline to catch."""
    rng = random.Random(seed)
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.txt"):
        old.unlink()

    base = datetime(2026, 9, 1).date()
    invoices = []
    for i in range(n):
        vendor, category = rng.choice(list(VENDORS.items()))
        lines = []
        for desc in rng.sample(CATALOG[category], k=rng.randint(1, 3)):
            lines.append((desc, rng.randint(1, 5),
                          round(rng.uniform(*PRICE_RANGE[category]), 2), 0))
        invoices.append(finalize(dict(
            vendor=vendor, lines=lines, rate=rng.choice([0.05, 0.08, 0.10]),
            inv_no=f"INV-{1001 + i}", date=base + timedelta(days=rng.randint(0, 60)),
            po=f"PO-{rng.randint(5000, 5999)}" if i % 3 == 0 else None, style=i % 3,
        )))

    # Deliberate problems
    invoices[4]["total"] = round(invoices[4]["total"] + 50, 2)      # math error
    invoices[7]["inv_no"] = None                                    # missing invoice number
    for j, (d, q, u, a) in enumerate(invoices[10]["lines"]):        # unusually large amount
        invoices[10]["lines"][j] = (d, q, u * 25, a)
    finalize(invoices[10])
    invoices.append(dict(invoices[2]))                              # exact duplicate

    for idx, inv in enumerate(invoices, start=1):
        (folder / f"invoice_{idx:03d}.txt").write_text(render_invoice(inv), encoding="utf-8")
    return len(invoices)


# --------------------------------------------------------------------------
# 2. Text extraction (txt / pdf / image)
# --------------------------------------------------------------------------
def extract_text(path):
    ext = path.suffix.lower()
    if ext == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise RuntimeError("install pdfplumber to read PDFs: pip install pdfplumber")
        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError("install pytesseract and pillow (plus the Tesseract binary) "
                               "to OCR images")
        return pytesseract.image_to_string(Image.open(path))
    raise RuntimeError(f"unsupported file type: {ext}")


# --------------------------------------------------------------------------
# 3. Field extraction
# --------------------------------------------------------------------------
MONEY = r"([\d,]+\.\d{2})"
LINE_RE = re.compile(
    r"^\s*(.+?)\s+(\d+)\s+[$€£₹]?\s*" + MONEY + r"\s+[$€£₹]?\s*" + MONEY + r"\s*$"
)
INVOICE_NO_RE = re.compile(r"invoice\s*(?:no\.?|number|#)\s*[:#]?\s*([A-Z0-9\-/]+)", re.I)
PO_RE = re.compile(r"\bPO\s*(?:number|no\.?|#)\s*[:#]?\s*([A-Z0-9\-]+)", re.I)
SUBTOTAL_RE = re.compile(r"\bsub[\s-]*total\b[^\n]*?" + MONEY, re.I)
TAX_RE = re.compile(r"\b(?:tax|vat|gst)\b[^\n]*?" + MONEY, re.I)
TOTAL_RE = re.compile(
    r"(?<!sub )(?<!sub-)(?<!sub)\b(?:grand\s+total|total\s+due|amount\s+due|total)\b[^\n]*?"
    + MONEY, re.I)
DATE_FORMATS = ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y",
                "%m/%d/%Y", "%d-%m-%Y")


def to_float(s):
    return float(s.replace(",", ""))


def parse_date(s):
    s = s.strip().rstrip(".")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def find_date(text, label):
    for line in text.splitlines():
        m = re.match(rf"\s*{label}\b\s*[:\-]?\s*(.+)$", line, re.I)
        if m:
            d = parse_date(m.group(1))
            if d:
                return d
    return None


def detect_currency(text):
    for symbol, code in (("€", "EUR"), ("£", "GBP"), ("₹", "INR"), ("$", "USD")):
        if symbol in text:
            return code
    return None


def parse_invoice(text):
    lines = [l for l in text.splitlines() if l.strip()]
    vendor = lines[0].strip() if lines else None
    if vendor and vendor.isupper():
        vendor = vendor.title()

    m = INVOICE_NO_RE.search(text)
    po = PO_RE.search(text)
    sub = SUBTOTAL_RE.search(text)
    tax = TAX_RE.search(text)
    totals = TOTAL_RE.findall(text)

    items = []
    for line in text.splitlines():
        lm = LINE_RE.match(line)
        if lm:
            items.append({"description": lm.group(1).strip(), "qty": int(lm.group(2)),
                          "unit_price": to_float(lm.group(3)), "amount": to_float(lm.group(4))})

    return {
        "vendor": vendor,
        "invoice_number": m.group(1) if m else None,
        "invoice_date": find_date(text, r"(?:invoice\s+)?date"),
        "due_date": find_date(text, r"(?:payment\s+)?due\s+date"),
        "po_number": po.group(1) if po else None,
        "currency": detect_currency(text),
        "subtotal": to_float(sub.group(1)) if sub else None,
        "tax": to_float(tax.group(1)) if tax else None,
        "total": to_float(totals[-1]) if totals else None,
        "line_items": items,
    }


# --------------------------------------------------------------------------
# 4. Validation
# --------------------------------------------------------------------------
def validate(rec):
    issues = []
    for field in ("vendor", "invoice_number", "invoice_date", "total"):
        if rec[field] in (None, ""):
            issues.append(f"missing_{field}")

    sub, tax, total = rec["subtotal"], rec["tax"], rec["total"]
    if None not in (sub, tax, total) and abs(sub + tax - total) > AMOUNT_TOLERANCE:
        issues.append("total_mismatch")
    if rec["line_items"] and sub is not None:
        if abs(sum(i["amount"] for i in rec["line_items"]) - sub) > AMOUNT_TOLERANCE:
            issues.append("line_items_mismatch")
    if rec["invoice_date"] and rec["due_date"] and rec["due_date"] < rec["invoice_date"]:
        issues.append("due_before_invoice_date")
    return issues


# --------------------------------------------------------------------------
# 5. Expense categorization (TF-IDF + logistic regression)
# --------------------------------------------------------------------------
def train_category_model():
    """Trained on the demo catalog. Replace with your own labeled invoices in production."""
    texts, labels = [], []
    for vendor, category in VENDORS.items():
        for item in CATALOG[category]:
            texts += [f"{vendor} {item}", item]
            labels += [category, category]
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
        LogisticRegression(max_iter=1000, C=10),
    )
    model.fit(texts, labels)
    return model


# --------------------------------------------------------------------------
# 6. Duplicate & anomaly detection
# --------------------------------------------------------------------------
def vendor_key(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def flag_duplicates(df):
    key = df["vendor"].map(vendor_key)
    tmp = df.assign(_vendor=key)
    same_number = (tmp.duplicated(["_vendor", "invoice_number"], keep="first")
                   & tmp["invoice_number"].notna())
    same_amount_date = (tmp.duplicated(["_vendor", "total", "invoice_date"], keep="first")
                        & tmp["total"].notna() & tmp["invoice_date"].notna())
    return same_number | same_amount_date


def flag_anomalies(df, z_cut=3.5, min_group=5):
    """Robust z-score (median/MAD) on log(total), computed within each category."""
    flags = pd.Series(False, index=df.index)
    for _, group in df.groupby("category"):
        vals = np.log1p(group["total"].dropna())
        if len(vals) < min_group:
            continue
        median = vals.median()
        mad = (vals - median).abs().median()
        if mad == 0:
            continue
        z = 0.6745 * (vals - median) / mad
        flags.loc[z[z.abs() > z_cut].index] = True
    return flags


# --------------------------------------------------------------------------
# 7. Routing
# --------------------------------------------------------------------------
def decide(row):
    if row["is_duplicate"]:
        return "REJECTED_DUPLICATE", "duplicate_invoice"
    reasons = [r for r in str(row["issues"]).split(";") if r]
    if row["category_confidence"] < MIN_CATEGORY_CONFIDENCE:
        reasons.append("low_category_confidence")
    if row["is_anomaly"]:
        reasons.append("unusual_amount_for_category")
    if pd.notna(row["total"]) and row["total"] > AUTO_APPROVE_LIMIT:
        reasons.append("above_auto_approve_limit")
    return ("NEEDS_REVIEW" if reasons else "AUTO_APPROVED"), ";".join(reasons)


# --------------------------------------------------------------------------
# 8. Pipeline
# --------------------------------------------------------------------------
def process_folder(folder):
    model = train_category_model()
    records, failed = [], []

    for path in sorted(p for p in folder.iterdir() if p.is_file()):
        try:
            rec = parse_invoice(extract_text(path))
        except Exception as exc:  # keep going if one file is unreadable
            failed.append((path.name, str(exc)))
            continue
        rec["file"] = path.name
        rec["issues"] = ";".join(validate(rec))
        rec["description"] = "; ".join(i["description"] for i in rec["line_items"])
        rec["n_items"] = len(rec["line_items"])
        del rec["line_items"]
        records.append(rec)

    if not records:
        raise SystemExit(f"No readable invoices found in {folder}. Failures: {failed}")

    df = pd.DataFrame(records)
    for col in ("subtotal", "tax", "total"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("invoice_date", "due_date"):
        df[col] = df[col].map(lambda d: d.isoformat() if d else None)

    # Categorize
    text_in = (df["vendor"].fillna("") + " " + df["description"].fillna("")).tolist()
    proba = model.predict_proba(text_in)
    df["category"] = model.classes_[proba.argmax(axis=1)]
    df["category_confidence"] = proba.max(axis=1).round(3)

    # Risk checks and routing
    df["is_duplicate"] = flag_duplicates(df)
    df["is_anomaly"] = flag_anomalies(df)
    decisions = df.apply(decide, axis=1, result_type="expand")
    df["status"], df["reasons"] = decisions[0], decisions[1]

    columns = ["file", "vendor", "invoice_number", "invoice_date", "due_date", "po_number",
               "currency", "n_items", "description", "subtotal", "tax", "total", "category",
               "category_confidence", "status", "reasons"]
    return df[columns], failed


def print_summary(df, failed):
    print(f"\nProcessed {len(df)} invoices" + (f" ({len(failed)} unreadable)" if failed else ""))
    print("\nStatus breakdown:")
    print(df["status"].value_counts().to_string())

    paid = df[df["status"] != "REJECTED_DUPLICATE"]
    print("\nSpend by category (duplicates excluded):")
    print(paid.groupby("category")["total"].agg(["count", "sum"])
          .rename(columns={"count": "invoices", "sum": "total_spend"})
          .round(2).sort_values("total_spend", ascending=False).to_string())

    queue = df[df["status"] != "AUTO_APPROVED"]
    if not queue.empty:
        print("\nNeeds attention:")
        print(queue[["file", "vendor", "invoice_number", "total", "status", "reasons"]]
              .to_string(index=False))
    for name, err in failed:
        print(f"  could not read {name}: {err}")


def main():
    parser = argparse.ArgumentParser(description="Intelligent Invoice Automation")
    parser.add_argument("--input", type=Path, default=None,
                        help="Folder of invoices (.txt/.pdf/.png/.jpg). "
                             "Omit to generate and use sample invoices.")
    parser.add_argument("--count", type=int, default=30,
                        help="Number of sample invoices to generate")
    args = parser.parse_args()

    folder = args.input
    if folder is None:
        total = generate_sample_invoices(SAMPLE_DIR, n=args.count)
        print(f"Generated {total} sample invoices in ./{SAMPLE_DIR}/ "
              "(includes a duplicate, a math error, a missing number and an outlier)")
        folder = SAMPLE_DIR
    elif not folder.is_dir():
        raise SystemExit(f"Input folder not found: {folder}")

    df, failed = process_folder(folder)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "processed_invoices.csv", index=False)
    df[df["status"] != "AUTO_APPROVED"].to_csv(OUTPUT_DIR / "review_queue.csv", index=False)

    print_summary(df, failed)
    print(f"\nSaved results to ./{OUTPUT_DIR}/processed_invoices.csv and review_queue.csv")


if __name__ == "__main__":
    main()
