import argparse
import json

from nlp_lab.config import ARTIFACT_DIR, KNOWLEDGE_PATH, ROOT
from nlp_lab.retrieval import KnowledgeRetriever


def evaluate(path=ROOT / "data" / "retrieval_eval.json", top_k: int = 3) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    retriever = KnowledgeRetriever(KNOWLEDGE_PATH)
    reciprocal_ranks, recall_1, recall_k, details = [], [], [], []
    for case in cases:
        results = retriever.search(case["query"], top_k=top_k)
        ids = [item["id"] for item in results]
        relevant = case["relevant_document"]
        rank = ids.index(relevant) + 1 if relevant in ids else None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        recall_1.append(float(rank == 1))
        recall_k.append(float(rank is not None))
        details.append({"query": case["query"], "relevant_document": relevant,
                        "retrieved": ids, "relevant_rank": rank})
    n = len(cases)
    return {"sample_count": n, "recall_at_1": sum(recall_1) / n,
            f"recall_at_{top_k}": sum(recall_k) / n,
            "mrr_at_k": sum(reciprocal_ranks) / n, "top_k": top_k,
            "details": details,
            "notice": "Conjunto sintético pequeno; mede apenas recuperação lexical, não qualidade de resposta LLM."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", action="store_true", help="Registra métricas no MLflow local")
    args = parser.parse_args()
    result = evaluate()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = ARTIFACT_DIR / "retrieval_metrics.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.tracking:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').resolve().as_posix()}")
        mlflow.set_experiment("nlp-engineering-lab")
        with mlflow.start_run(run_name="tfidf-retrieval-evaluation"):
            mlflow.log_params({"top_k": result["top_k"], "sample_count": result["sample_count"]})
            mlflow.log_metrics({key: value for key, value in result.items()
                                if isinstance(value, (int, float)) and key != "top_k"})
            mlflow.log_artifact(str(output_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
