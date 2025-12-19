.PHONY: help install build-summary run-summary build-fulltext run-fulltext

help:
	@echo "Targets:"
	@echo "  install        - install deps + editable package"
	@echo "  run-summary    - run full pipeline in summary mode"
	@echo "  run-fulltext   - run full pipeline in fulltext mode (requires data/full_text/)"

install:
	pip install -r requirements.txt
	pip install -e .

run-summary:
	python -m nlp_headline_body_sentiment.pipeline run-all --mode summary

run-fulltext:
	python -m nlp_headline_body_sentiment.pipeline run-all --mode fulltext


