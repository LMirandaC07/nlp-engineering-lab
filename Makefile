.PHONY: install train serve test mlflow
install:
	python -m pip install -e ".[dev,mcp]"
train:
	python -m nlp_lab.train
serve:
	uvicorn nlp_lab.api:app --reload
test:
	pytest
mlflow:
	mlflow ui --backend-store-uri ./mlruns
