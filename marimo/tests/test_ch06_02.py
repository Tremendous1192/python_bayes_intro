# - 作成日: 2026-09-27
# - 目的: 効果検証の標準化・座標・縮小事前分布・予測を検査する。
# - 役割: Notebook専用の本実行検証。
# - 使用: uv run --locked --group notebook python tests/test_ch06_02.py
# - 制約: 新規プロセス、CPU逐次推論、ローカルデータ。
# - 非対応: GUI・EXE・因果関係の保証。
from types import SimpleNamespace

import sys
from pathlib import Path

# - 直接実行とWindows子プロセスで、移動先から同じNotebookを読み込む。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from ch06_02_effects import app


def test_effects() -> None:
    """- 列名と数式を独立したNumPy計算で検査する。
    - 引数・戻り値: なし。前提: データ準備済み。
    - 副作用: 推論・描画・英語出力。失敗: 元の例外を伝える。
    - 使用例: test_effects()。
    """
    print("Checking regression effects: book", flush=True)
    # - 不一致を無視せず失敗を報告する。
    try:
        _, values = app.run(defs={"sampling_mode": SimpleNamespace(value="book"),
                                 "run_inference": SimpleNamespace(value=True)})
        assert values["X"].shape == (101, 10)
        assert "score" in values["df1"] and "score" not in values["X"]
        np.testing.assert_allclose(values["X"].mean(), 0, atol=1e-14)
        np.testing.assert_allclose(values["X"].std(ddof=1), 1, atol=1e-14)
        # - 入力行列と保存した係数座標が両モデルで同一であることを確認する。
        for name in ("idata1", "idata2"):
            result = values[name]
            posterior = result["posterior"]
            np.testing.assert_array_equal(posterior["predictors"], values["columns"])
            assert posterior["alpha"].shape == (4, 1000, 10)
            assert (posterior["epsilon"] > 0).all()
            alpha = posterior["alpha"].mean(("chain", "draw")).values
            beta = float(posterior["beta"].mean())
            actual = posterior["mu"].mean(("chain", "draw")).values
            np.testing.assert_allclose(actual, values["X"].to_numpy() @ alpha + beta, atol=1e-10)
            assert np.isfinite(actual).all()
            print(f"{name}: divergences={int(result['sample_stats']['diverging'].sum())}", flush=True)
            # - 縮小モデルの受容率調整が発散を解消したことを本実行で確認する。
            assert int(result["sample_stats"]["diverging"].sum()) == 0
        p = values["idata2"]["posterior"]
        # - 名前で選んだ4変数を同じ順序で数式の独立検算に渡す。
        tau, lam, c2, z = (p[key].values for key in ("tau", "lam", "c2", "z"))
        expected = z * tau[..., None] * lam * np.sqrt(
            c2[..., None] / (c2[..., None] + tau[..., None]**2 * lam**2))
        np.testing.assert_allclose(p["alpha"], expected, rtol=1e-12, atol=1e-12)
        assert not matplotlib.pyplot.get_fignums()
    # - 原因のスタックトレースを失わず出力する。
    except Exception:
        print("Failed: regression effects\n", flush=True)
        raise
    print("Passed: regression effects\n", flush=True)


# - インポートでは推論を開始しない。
if __name__ == "__main__":
    test_effects()
