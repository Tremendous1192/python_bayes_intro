# - 作成日: 2026-09-27
# - 目的: ABテストの差の符号と確率を独立したベータ積分で検証する。
# - 役割: Notebook専用の本実行検証。
# - 使用: uv run --locked --group notebook python tests/test_ch06_01.py
# - 制約: 新規プロセス、CPU逐次実行、外部接続なし。
# - 非対応: GUI・EXE・元版との乱数列の完全一致。
from types import SimpleNamespace

import sys
from pathlib import Path

# - 直接実行とWindows子プロセスで、移動先から同じNotebookを読み込む。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy import integrate, stats

from ch06_01_ab_test import app


def test_ab_probability() -> None:
    """- 解析的な積分を基準にMCMCと直接標本化を検査する。
    - 引数・戻り値: なし。前提: 固定環境。副作用: 推論・描画・英語出力。
    - 失敗: AssertionError等を伝える。使用例: test_ab_probability()。
    """
    print("Checking A/B probabilities against exact Beta integrals", flush=True)
    # - 数値的不一致や実行例外を失敗として表示する。
    try:
        _, values = app.run(defs={"sampling_mode": SimpleNamespace(value="book"),
                                 "run_inference": SimpleNamespace(value=True)})
        # - Aの密度×Bの累積分布を積分するとP(A>B)になる。
        for suffix, a, b in [("s", (3, 39), (3, 24)), ("y", (61, 1141), (111, 1491))]:
            def integrand(x: float) -> float:
                """- 確率積分の被積分値。引数: 率x。戻り値: 密度積。
                - 前提: xは0以上1以下。副作用: なし。失敗: SciPy例外。
                - 使用例: integrand(0.1)。
                """
                return float(stats.beta.pdf(x, *a) * stats.beta.cdf(x, *b))
            expected, error = integrate.quad(integrand, 0, 1, epsabs=1e-10)
            assert error < 1e-8
            assert abs(values[f"n1_rate_{suffix}"] - expected) < 0.035
            assert abs(values[f"n1_rate_{suffix}2"] - expected) < 0.02
            posterior = values[f"idata_{suffix}"]["posterior"]
            np.testing.assert_allclose(posterior[f"delta_prob_{suffix}"],
                                       posterior[f"p_{suffix}_b"] - posterior[f"p_{suffix}_a"])
            assert posterior.sizes["chain"] == 4 and posterior.sizes["draw"] == 1000
            assert values[f"samples_{suffix}2"]["prior"].sizes["draw"] == 10000
            print(f"{suffix}: exact P(A > B) = {expected:.6f}", flush=True)
        assert not matplotlib.pyplot.get_fignums()
    # - 例外は保持して呼出元へ伝える。
    except Exception:
        print("Failed: A/B probabilities\n", flush=True)
        raise
    print("Passed: A/B probabilities\n", flush=True)


# - インポートだけでは重い推論を開始しない。
if __name__ == "__main__":
    test_ab_probability()
