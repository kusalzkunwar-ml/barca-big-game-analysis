"""Train an expected-goals model on non-penalty shots and benchmark it against StatsBomb's xG.

Matches are held out as whole groups (GroupKFold) so shots from the same game never sit in
both train and test. Out-of-fold predictions are written to data/processed/shots_xg.parquet.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"

NUMERIC = ["distance", "angle", "log_distance", "defenders_in_cone", "gk_distance"]
FLAGS = ["is_header", "first_time", "under_pressure"]
CATEGORICAL = ["shot_type", "play_pattern", "technique"]


def features(shots: pd.DataFrame) -> pd.DataFrame:
    df = shots[shots.shot_type != "Penalty"].copy()
    df["log_distance"] = np.log1p(df.distance)
    df["is_header"] = df.body_part == "Head"
    df["defenders_in_cone"] = df.defenders_in_cone.fillna(df.defenders_in_cone.median())
    df["gk_distance"] = df.gk_distance.fillna(df.distance)   # no keeper in the frame: treat as unguarded line
    df[FLAGS] = df[FLAGS].astype(float)
    return df


def models():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC + FLAGS),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=30), CATEGORICAL),
    ])
    return {
        "logistic": make_pipeline(prep, LogisticRegression(max_iter=2000, C=1.0)),
        "gradient_boosting": make_pipeline(
            prep, HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                 l2_regularization=1.0, random_state=0)),
    }


def scores(y, p) -> dict:
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "roc_auc": roc_auc_score(y, p),
            "xg_total": p.sum(), "goals": y.sum()}


def main():
    df = features(pd.read_parquet(DATA / "shots.parquet"))
    X, y, groups = df[NUMERIC + FLAGS + CATEGORICAL], df.goal.astype(int).values, df.match_id.values
    results = {"statsbomb_xg": scores(y, df.xg_statsbomb.values)}
    for name, model in models().items():
        oof = np.zeros(len(df))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            model.fit(X.iloc[tr], y[tr])
            oof[te] = model.predict_proba(X.iloc[te])[:, 1]
        df[f"xg_{name}"] = oof
        results[name] = scores(y, oof)
    table = pd.DataFrame(results).T
    print(f"{len(df):,} non-penalty shots, {y.sum():,} goals\n")
    print(table.round(4))
    df.to_parquet(DATA / "shots_xg.parquet", index=False)
    table.to_csv(DATA / "xg_model_scores.csv")


if __name__ == "__main__":
    main()
