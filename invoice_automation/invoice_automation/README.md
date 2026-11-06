# Intelligent Invoice Automation

Reads invoices, extracts key fields, validates them, categorizes the expense, flags duplicates and unusual amounts, and routes each invoice for approval or review.

## Setup

```bash
pip install -r requirements.txt
```

For real PDFs or scanned images, uncomment the optional packages in `requirements.txt`. Images also need the Tesseract OCR binary installed.

## Run

```bash
# Generate sample invoices and process them
python invoice_automation.py

# Process your own folder (.txt, .pdf, .png, .jpg)
python invoice_automation.py --input my_invoices/
```

## Pipeline

1. Extract text from txt, PDF or image files
2. Extract fields with regex: vendor, invoice number, dates, PO number, subtotal, tax, total, line items
3. Validate: missing fields, subtotal + tax vs total, line items vs subtotal, date order
4. Categorize with TF-IDF and logistic regression
5. Detect duplicates and unusual amounts (robust z-score within each category)
6. Route: `AUTO_APPROVED`, `NEEDS_REVIEW` or `REJECTED_DUPLICATE`

## Outputs (in `outputs/`)

- `processed_invoices.csv` - all invoices with extracted fields, category, status and reasons
- `review_queue.csv` - invoices that need a human

## Settings (top of `invoice_automation.py`)

- `AUTO_APPROVE_LIMIT` - invoices above this amount always go to review
- `MIN_CATEGORY_CONFIDENCE` - below this, the predicted category is not trusted
- `AMOUNT_TOLERANCE` - allowed rounding difference when checking totals

## Note

The categorizer is trained on a small built-in demo catalog. Replace `train_category_model()` with your own labeled invoices before using real data.
