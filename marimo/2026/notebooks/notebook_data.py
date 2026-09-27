# - 作成日: 2026-09-27
# - 目的: 各章で共用する教材データの版と型を確認して読む。
# - 役割: 複数Notebookでの取得・型・検査の食い違いを防ぐNotebook専用処理。
# - 使用: load_data("iris.csv")。同梱CSVを使用し、再取得はlog/prepare_data.pyで行う。
# - 制約: data/sources.jsonの3本のみ。読取専用、通信なし。
# - 非対応: 任意CSV、依存導入、EXE配布。
import hashlib
import json
from pathlib import Path

import pandas as pd

# - 入力は保存済みCSV、出力は明示した型のDataFrame。
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def load_data(name: str) -> pd.DataFrame:
    """固定データを読み込む。引数はCSV名、戻り値は独立したDataFrame。
    前提: 初回取得済み。副作用なし。欠落はFileNotFoundError、不整合はValueError。
    例: load_data("iris.csv")。元データとキャッシュを書き換えない。
    """
    sources = json.loads((DATA_ROOT / "sources.json").read_text(encoding="utf-8"))
    # - 任意パスや未知のデータを読ませない。
    if name not in sources or Path(name).name != name:
        raise ValueError("Unknown teaching dataset.")
    path = DATA_ROOT / name
    # - オフライン時も操作方法を英語で示し、自動取得しない。
    if not path.is_file():
        raise FileNotFoundError("Run python log/prepare_data.py --download from the project root.")
    # - 同じ名前の別データを推論に投入しない。
    if hashlib.sha256(path.read_bytes()).hexdigest() != sources[name]["sha256"]:
        raise ValueError(f"Checksum mismatch: {name}")
    # - Irisの数値列とカテゴリ列を明示する。
    if name == "iris.csv":
        frame = pd.read_csv(path, dtype={"sepal_length": "float64", "sepal_width": "float64",
                                        "petal_length": "float64", "petal_width": "float64",
                                        "species": "str"})
        # - 分布比較の前提である3種各50件を確認する。
        if frame.shape != (150, 5) or frame.isna().any().any():
            raise ValueError("Invalid Iris shape or missing values.")
    # - IRTは先頭列を受験者ID、それ以外を0/1整数として読む。
    elif name == "irt-sample.csv":
        frame = pd.read_csv(path, index_col=0)
        frame = frame.astype("int64")
        # - 二値観測と座標の一意性を検査する。
        if frame.shape != (1000, 50) or not frame.isin([0, 1]).all().all() or not frame.index.is_unique:
            raise ValueError("Invalid IRT observations or coordinates.")
    # - 回帰データは欠損を保持し、除去と標準化を学習セルで説明する。
    else:
        frame = pd.read_csv(path, index_col=0).astype("float64")
        # - 目的変数の欠落では後続の推論を開始しない。
        if "score" not in frame or frame.empty:
            raise ValueError("Missing regression target.")
    return frame
