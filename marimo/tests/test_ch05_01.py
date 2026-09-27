# - 作成日: 2026-09-27
# - 目的: 5.1節のデータ抽出・推定値・精度変換を検証する。
# - 役割: Notebook専用の新規セッション検査。
# - 使用: uv run --locked --group notebook python tests/test_ch05_01.py
# - 制約: Book runを実行。固定入力、同時推論1。
# - 非対応: 画面操作・EXE配布。
from types import SimpleNamespace

import sys
from pathlib import Path

# - 直接実行とWindows子プロセスで、移動先から同じNotebookを読み込む。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from ch05_01_distribution import app


def test_distribution() -> None:
    """モデル3個を検査する。引数・戻り値なし。前提: 固定環境とIris。
    副作用は推論・描画・英語出力。不一致はAssertionError。例: test_distribution()。
    """
    print("Checking chapter 5.1: book", flush=True)
    # - 例外と数値不一致を検査失敗として扱う。
    try:
        _, values = app.run(defs={"sampling_mode": SimpleNamespace(value="book"),
                                 "run_inference": SimpleNamespace(value=True)})
        assert len(values["X"]) == 50 and len(values["X_less"]) == 5
        assert abs(values["mu_mean1"] - 5.006) < 0.1
        assert abs(values["sigma_mean1"] - 0.3524896872) < 0.1
        # - 全モデルで事後分布の支持・有限性・保存反復数を確認する。
        for name in ("idata1", "idata2", "idata3"):
            posterior = values[name]["posterior"]
            assert posterior.sizes["draw"] == 1000 and posterior.sizes["chain"] == 4
            assert np.isfinite(posterior["mu"]).all()
            assert (posterior["sigma"] > 0).all()
        posterior = values["idata3"]["posterior"]
        np.testing.assert_allclose(posterior["sigma"] ** 2 * posterior["tau"], 1, rtol=1e-12)
        assert float(values["idata2"]["posterior"]["mu"].std()) > float(values["idata1"]["posterior"]["mu"].std())
        assert int(values["idata2"]["sample_stats"]["diverging"].sum()) == 0
    # - 失敗時も元の原因を残す。
    except Exception:
        print("Failed: chapter 5.1\n", flush=True)
        raise
    print("Passed: chapter 5.1\n", flush=True)


# - import時に推論を実行しない。
if __name__ == "__main__":
    test_distribution()
