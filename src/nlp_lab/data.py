import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(data.columns):
        raise ValueError(f"Dataset precisa conter as colunas: {sorted(required)}")
    data = data[["text", "label"]].dropna().copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()
    data = data[(data.text != "") & (data.label != "")]
    data["normalized"] = data.text.str.casefold().str.replace(r"\s+", " ", regex=True)
    duplicated = data.duplicated("normalized", keep=False)
    conflicting = data.loc[duplicated].groupby("normalized").label.nunique().gt(1)
    if conflicting.any():
        raise ValueError("Há textos duplicados associados a classes diferentes.")
    return data.drop_duplicates("normalized").drop(columns="normalized").reset_index(drop=True)
