import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nlp_lab.config import KNOWLEDGE_PATH


class KnowledgeRetriever:
    """Recuperação lexical local, pequena e auditável; não é um embedding semântico."""
    def __init__(self, path: Path = KNOWLEDGE_PATH, threshold: float = 0.08):
        self.documents = json.loads(path.read_text(encoding="utf-8"))
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self.matrix = self.vectorizer.fit_transform(
            [f"{d['title']} {d['text']}" for d in self.documents])
        self.threshold = threshold

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        indices = scores.argsort()[::-1][:top_k]
        return [{**self.documents[i], "score": float(scores[i])} for i in indices
                if scores[i] >= self.threshold]

    def answer(self, query: str) -> dict:
        hits = self.search(query)
        if not hits:
            return {"answer": "Não encontrei informação suficiente na base demonstrativa.",
                    "sources": [], "grounded": False}
        cited = hits[0]
        return {"answer": cited["text"], "sources": [cited["id"]],
                "grounded": True, "retrieval_score": cited["score"]}
