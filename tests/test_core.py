import pandas as pd
from fastapi.testclient import TestClient

from nlp_lab.agent import SupportAgent
from nlp_lab.api import app
from nlp_lab.data import load_dataset
from nlp_lab.eval_rag import evaluate
from nlp_lab.retrieval import KnowledgeRetriever


def test_dataset_is_unique_and_stratifiable():
    data = load_dataset("data/intents.csv")
    assert data.text.is_unique
    assert data.label.nunique() == 8
    assert data.groupby("label").size().min() >= 10


def test_dataset_rejects_conflicting_duplicate(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["Oi", "oi"], "label": ["a", "b"]}).to_csv(path, index=False)
    try:
        load_dataset(str(path))
    except ValueError as exc:
        assert "duplicados" in str(exc)
    else:
        raise AssertionError("Esperava rejeição de classes conflitantes")


def test_retrieval_cites_source_or_abstains():
    result = KnowledgeRetriever().answer("a clínica atende em feriados?")
    assert "grounded" in result
    if result["grounded"]:
        assert result["sources"]


def test_retrieval_evaluation_is_reproducible():
    results = evaluate()
    assert results["sample_count"] == 6
    assert 0.0 <= results["recall_at_3"] <= 1.0
    assert len(results["details"]) == 6


def test_agent_escalates_emergency():
    result = SupportAgent().run("meu cachorro está convulsionando", "emergencia")
    assert result["route"] == "human_or_emergency"
    assert result["tool"] is None


def test_api_health_and_validation():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        prediction = client.post("/predict", json={"text": "Quero marcar consulta para meu gato"})
        assert prediction.status_code == 200
        assert prediction.json()["label"] == "agendamento"
        assert "não é probabilidade" in prediction.json()["score_semantics"]
        assert client.post("/agent", json={"text": "marcar consulta"}).status_code == 200
        assert client.post("/agent", json={"text": " "}).status_code == 422


def test_llm_route_is_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with TestClient(app) as client:
        response = client.post("/agent/llm", json={"text": "Quais os horários?"})
        assert response.status_code == 503
