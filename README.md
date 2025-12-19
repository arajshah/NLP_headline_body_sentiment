# Gun Violence Media Framing Analysis

This repository is a **reproducible NLP pipeline** for analyzing **media framing in gun-violence news** by quantifying how **headline sentiment differs from body sentiment** (the *headline–body sentiment gap*), and how that gap varies across **annotation dimensions** and (optionally) across **publishers/outlets**.

## What it does

- Builds a **canonical article dataset** from `data/GVFC_extension_multimodal.csv` with consistent columns and validation.
- Computes sentiment for **headlines** and **bodies** (transformer model with long-text token chunking; optional VADER baseline).
- Computes `sent_gap = sent_head - sent_body`, writes **statistics + subgroup tables**, and saves figures + a short markdown report.
- Optionally merges **publisher domains** with an external **bias table** (if you provide one locally).

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m nlp_headline_body_sentiment.pipeline run-all --mode summary
```

Optional robustness baseline:

```bash
python -m nlp_headline_body_sentiment.pipeline run-all --mode summary --vader
```

## Outputs

- **Derived datasets**: `data/derived/`
- **Figures**: `reports/figures/<mode>/`
- **Report**: `reports/report_<mode>.md`

## Optional inputs

- **Scraped full text** (for `--mode fulltext`): `data/full_text/{id}.txt`
- **External bias table** (optional): pass `--bias-csv /path/to/bias.csv` (see `data/external/README.md`)


