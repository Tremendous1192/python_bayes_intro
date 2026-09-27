# - 作成日: 2026-09-27
# - 目的: 回帰の入力対応、事後予測、階層のカテゴリ対応、Excel保存を検査する。
# - 役割: Notebook専用の本実行検証。
# - 使用: uv run --locked --group notebook python tests/test_ch05_02_03.py
# - 制約: 各Notebook1800秒、同時推論1。固定データとローカル出力を使用。
# - 非対応: 画面操作・EXE配布。未調整例の収束成功は要求しない。
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


# - 直接実行とWindows子プロセスで、移動先から同じNotebookを読み込む。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd


def test_regression(module: str) -> None:
    """指定章を検査する。引数はモジュール名、戻り値なし。前提: データ準備済み。
    副作用は推論・描画・Excel保存・英語出力。不一致はAssertionError。
    例: test_regression("ch05_03_hierarchical")。
    """
    print(f"Checking {module}: book", flush=True)
    # - 推論や保存の例外を正常終了にしない。
    try:
        overrides = {"sampling_mode": SimpleNamespace(value="book"),
                     "run_inference": SimpleNamespace(value=True)}
        # - Excel保存もプロジェクト内の新規ファイルだけで検査する。
        if module == "ch05_03_hierarchical":
            overrides["export_table"] = SimpleNamespace(value=True)
        _, values = importlib.import_module(module).app.run(defs=overrides)
        # - 通常回帰は独立した最小二乗予測との整合を確認する。
        if module == "ch05_02_regression":
            assert values["sample_indexes"] == [40, 7, 1]
            assert values["model2"]["X_data"].get_value().shape == (50,)
            assert values["model3"]["X_data"].get_value().shape == (3,)
            design = np.column_stack([values["X"], np.ones(50)])
            expected = design @ np.linalg.lstsq(design, values["Y"], rcond=None)[0]
            actual = values["idata2"]["posterior"]["mu"].mean(("chain", "draw")).values
            assert np.sqrt(np.mean((actual - expected) ** 2)) < 0.15
            # - 調整・未調整の両方でモデルの支持と観測数を保持する。
            for name in ("idata2", "idata3", "idata4"):
                posterior = values[name]["posterior"]
                assert np.isfinite(posterior["alpha"]).all()
                assert (posterior["epsilon"] > 0).all()
                assert posterior.sizes["chain"] == 4 and posterior.sizes["draw"] == 1000
        # - 階層回帰は順序固定した3種と9観測の対応を確認する。
        else:
            np.testing.assert_array_equal(values["cl"], [0, 0, 0, 1, 1, 1, 2, 2, 2])
            np.testing.assert_allclose(values["MU"], [0.1] * 3 + [0.2] * 3 + [0.3] * 3)
            assert values["alpha_means"].shape == (3,)
            posterior = values["idata1"]["posterior"]
            predicted = posterior["mu"].mean(("chain", "draw")).values
            assert predicted.shape == (9,) and np.isfinite(predicted).all()
            assert np.sqrt(np.mean((predicted - values["Y"]) ** 2)) < 0.5
            assert "X_data" not in values["model2"].named_vars
            assert "X_data" in values["model3"].named_vars
            path = values["excel_path"]
            assert path.is_relative_to(Path(__file__).resolve().parents[1] / ".cache")
            saved = pd.read_excel(path, index_col=0, engine="openpyxl")
            assert saved.shape == (5, 5)
            np.testing.assert_allclose(saved["sepal_length"], values["df"].head()["sepal_length"])
    # - 失敗の詳細を保持して伝える。
    except Exception:
        print(f"Failed: {module}\n", flush=True)
        raise
    print(f"Passed: {module}\n", flush=True)


# - 各Notebookを前の状態に依存しないプロセスで実行する。
if __name__ == "__main__":
    # - 子プロセスでは1章だけを扱う。
    if len(sys.argv) == 2:
        test_regression(sys.argv[1])
    # - 親プロセスは同時推論を避けて順番に実行する。
    else:
        # - 階層モデルを含む各ケースに30分の上限を設ける。
        for module in ("ch05_02_regression", "ch05_03_hierarchical"):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), module],
                           check=True, timeout=1800)
