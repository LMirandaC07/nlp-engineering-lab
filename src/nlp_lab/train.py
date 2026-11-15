import json
import time
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nlp_lab.config import (
    ARTIFACT_DIR,
    DATA_PATH,
    ERRORS_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
)
from nlp_lab.data import load_dataset


def train(data_path: Path = DATA_PATH, artifact_dir: Path = ARTIFACT_DIR,
          tracking: bool = True) -> dict:
    data = load_dataset(str(data_path))
    x_train, x_holdout, y_train, y_holdout = train_test_split(
        data.text, data.label, test_size=0.30, stratify=data.label,
        random_state=RANDOM_STATE)
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_holdout, y_holdout, test_size=0.50, stratify=y_holdout,
        random_state=RANDOM_STATE)
    # O conjunto de validação é reservado para decisões futuras; o teste só mede o resultado final.
    _ = (x_valid, y_valid)
    candidates = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "multinomial_nb": MultinomialNB(),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "linear_svm": LinearSVC(class_weight="balanced"),
    }
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if tracking:
        mlflow.set_tracking_uri(f"sqlite:///{(artifact_dir.parent / 'mlflow.db').resolve().as_posix()}")
        mlflow.set_experiment("nlp-engineering-lab")
    rows, fitted = [], {}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                                       strip_accents="unicode")),
                             ("classifier", estimator)])
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed_ms = (time.perf_counter() - started) * 1000
        valid_pred = pipeline.predict(x_valid)
        test_pred = pipeline.predict(x_test)
        vp, vr, vf, _ = precision_recall_fscore_support(y_valid, valid_pred,
                                                       average="macro", zero_division=0)
        tp, tr, tf, _ = precision_recall_fscore_support(y_test, test_pred,
                                                       average="macro", zero_division=0)
        metrics = {"validation_accuracy": float(accuracy_score(y_valid, valid_pred)),
                   "validation_precision_macro": float(vp), "validation_recall_macro": float(vr),
                   "validation_macro_f1": float(vf),
                   "test_accuracy": float(accuracy_score(y_test, test_pred)),
                   "test_precision_macro": float(tp), "test_recall_macro": float(tr),
                   "test_macro_f1": float(tf), "fit_latency_ms": float(elapsed_ms)}
        rows.append({"model": name, **metrics})
        fitted[name] = pipeline
        if tracking:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "random_state": RANDOM_STATE,
                                   "train_rows": len(x_train), "test_rows": len(x_test),
                                   "tfidf_ngram_range": "1,2"})
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(pipeline, artifact_path="model")
    results = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    # Seleção usa validação; o teste fica separado para a comparação final.
    winner = str(results[results.model != "dummy_most_frequent"].iloc[0].model)
    model = fitted[winner]
    pred = model.predict(x_test)
    labels = sorted(data.label.unique())
    errors = pd.DataFrame({"text": x_test.to_list(), "expected": y_test.to_list(),
                           "predicted": pred})
    errors = errors[errors.expected != errors.predicted]
    errors.to_csv(artifact_dir / ERRORS_PATH.name, index=False)
    results.to_csv(artifact_dir / "comparison.csv", index=False)
    payload = {"selected_model": winner, "random_state": RANDOM_STATE,
               "rows_total": len(data), "rows_train": len(x_train),
               "rows_validation": len(x_valid), "rows_test": len(x_test),
               "class_distribution": data.label.value_counts().sort_index().to_dict(),
               "results": rows,
               "classification_report": classification_report(y_test, pred, labels=labels,
                                                              output_dict=True,
                                                              zero_division=0),
               "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
               "labels": labels,
               "selection_metric": "validation_macro_f1",
               "test_set_notice": "Pequeno dataset sintético; não representa desempenho em produção."}
    (artifact_dir / METRICS_PATH.name).write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    joblib.dump(model, artifact_dir / MODEL_PATH.name)
    if tracking:
        with mlflow.start_run(run_name="evaluation-artifacts"):
            for artifact in ("comparison.csv", ERRORS_PATH.name, METRICS_PATH.name,
                             MODEL_PATH.name):
                mlflow.log_artifact(str(artifact_dir / artifact))
    return payload


def main() -> None:
    result = train()
    print(json.dumps({"selected_model": result["selected_model"],
                      "results": result["results"], "artifacts": str(ARTIFACT_DIR)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
