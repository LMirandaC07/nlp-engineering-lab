from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "intents.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "intent_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
ERRORS_PATH = ARTIFACT_DIR / "error_analysis.csv"
KNOWLEDGE_PATH = ROOT / "data" / "knowledge_base.json"
RANDOM_STATE = 42
