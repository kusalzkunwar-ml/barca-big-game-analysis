"""Run every named query in sql/queries.sql against the processed parquet files.

Each result is printed and saved to outputs/<name>.csv. Import `connect` / `load_queries`
to reuse them from the notebook.
"""
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW matches AS SELECT * FROM '{(DATA / 'matches.parquet').as_posix()}'")
    con.execute(f"CREATE VIEW team_match AS SELECT * FROM '{(DATA / 'team_match.parquet').as_posix()}'")
    con.execute("CREATE VIEW barca AS SELECT * FROM team_match WHERE is_barca")
    con.execute("CREATE VIEW opp AS SELECT * FROM team_match WHERE NOT is_barca")
    return con


def load_queries() -> dict:
    text = (ROOT / "sql" / "queries.sql").read_text(encoding="utf-8")
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def main():
    con = connect()
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)
    for name, sql in load_queries().items():
        df = con.execute(sql).df()
        df.to_csv(out / f"{name}.csv", index=False)
        print(f"\n== {name} ==\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
