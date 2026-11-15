from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from nlp_lab.agent import SupportAgent
from nlp_lab.config import MODEL_PATH
from nlp_lab.gemini_agent import GeminiToolAgent


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("O texto não pode conter apenas espaços.")
        return value


class AgentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("O texto não pode conter apenas espaços.")
        return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    app.state.agent = SupportAgent()
    app.state.genai_agent = GeminiToolAgent.from_environment()
    yield


app = FastAPI(title="NLP Engineering Lab", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(request: PredictRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ausente. Execute python -m nlp_lab.train.")
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="O texto não pode conter apenas espaços.")
    prediction = str(model.predict([text])[0])
    classifier = model.named_steps["classifier"]
    scores = model.decision_function([text])[0] if hasattr(classifier, "decision_function") else model.predict_proba([text])[0]
    labels = list(model.classes_)
    return {"label": prediction, "scores": {label: float(score) for label, score in zip(labels, scores)},
            "score_semantics": "decision_function (não é probabilidade)" if hasattr(classifier, "decision_function") else "probabilidade estimada"}


@app.post("/agent")
def agent(request: AgentRequest):
    model = getattr(app.state, "model", None)
    intent = str(model.predict([request.text])[0]) if model is not None else None
    return app.state.agent.run(request.text, intent)


@app.post("/agent/llm")
def llm_agent(request: AgentRequest):
    service = getattr(app.state, "genai_agent", None)
    if service is None:
        raise HTTPException(status_code=503,
                            detail="Configure GEMINI_API_KEY e GEMINI_MODEL no ambiente para habilitar o agente.")
    from google.genai.errors import APIError

    try:
        return service.run(request.text)
    except (APIError, TimeoutError, ConnectionError, RuntimeError) as exc:
        # Não devolve detalhes do provider, headers ou credenciais ao cliente HTTP.
        raise HTTPException(status_code=502,
                            detail=f"Falha ao consultar o provider GenAI ({type(exc).__name__}).") from None
