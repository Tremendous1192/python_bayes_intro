# - 作成日: 2026-09-27
# - 目的: 第2章の分布と区間を解析解・独立したSciPy実装で確認する。
# - 役割: Notebook専用の数値回帰検査。
# - 使用: uv run --locked --group notebook python notebooks/test_ch02.py
# - 制約: 固定データ準備済み。各Notebookは新規プロセスで順次実行。
# - 非対応: 画面操作・EXE配布。
import importlib
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, stats

# - 章内の学習単位ごとに検証プロセスを分離する。
MODULES = ("ch02_discrete", "ch02_normal", "ch02_uniform_beta", "ch02_intervals")


def test_chapter(module: str) -> None:
    """指定Notebookを検証する。引数はMODULESの名前、戻り値なし。
    前提: 固定環境。副作用はセル実行・英語出力。不一致はAssertionError。
    例: test_chapter("ch02_intervals")。図はAggで描く。
    """
    print(f"Checking {module}", flush=True)
    # - 正常と失敗を明確に分ける。
    try:
        _, values = importlib.import_module(module).app.run()
        # - 離散分布の理論平均・分散からモンテカルロ誤差の上限を求める。
        if module == "ch02_discrete":
            expected = [(0.5, 0.25), (2.5, 1.25), (25, 12.5)]
            # - 分布ごとに独立した理論値と比較する。
            for prior, (mean, variance) in zip(values["priors"].values(), expected):
                sample = prior["prior"]["x"].values
                assert sample.shape == (1, 500)
                assert abs(sample.mean() - mean) < 5 * np.sqrt(variance / 500)
        # - 正規分布はシードを共有した位置・尺度変換も確認する。
        elif module == "ch02_normal":
            first, second = [p["prior"]["x"].values for p in values["priors"].values()]
            np.testing.assert_allclose(second, first * 2 + 3)
            assert len(values["setosa"]) == 50
        # - ベータ密度は元の例に加え、旧式が誤るパラメータで積分する。
        elif module == "ch02_uniform_beta":
            # - 正規化の回帰を、実装と別の積分法で検出する。
            for alpha, beta in [(2, 2), (3, 4), (21, 31)]:
                mass = integrate.quad(lambda p: values["Beta"](p, alpha, beta), 0, 1)[0]
                assert abs(mass - 1) < 1e-9
                assert abs(values["Beta"](0.4, alpha, beta) - stats.beta.pdf(0.4, alpha, beta)) < 1e-10
            assert np.all(values["priors"]["HalfNormal sigma=1"]["prior"]["x"].values >= 0)
            assert np.all((values["priors"]["Uniform 0.1 to 0.9"]["prior"]["x"].values >= 0.1)
                          & (values["priors"]["Uniform 0.1 to 0.9"]["prior"]["x"].values <= 0.9))
        # - 中央区間の質量とHDI端点の等密度を確認する。
        else:
            assert abs(values["ci_mass"] - 0.8) < 1e-12
            np.testing.assert_allclose(stats.chi2.pdf(values["hdi_bounds"], 3), 0.05, atol=1e-10)
            assert 0 < values["hdi_mass"] < 1
        assert plt.get_fignums() == []
    # - 数値・セルの失敗は理由を保持して再送出する。
    except Exception:
        print(f"Failed: {module}\n", flush=True)
        raise
    print(f"Passed: {module}\n", flush=True)


# - 前のNotebookの状態を使わず、各ケースを期限付きで実行する。
if __name__ == "__main__":
    # - 子プロセスは指定したNotebookだけを処理する。
    if len(sys.argv) == 2:
        test_chapter(sys.argv[1])
    # - 親プロセスはCPU競合を避けて順番に起動する。
    else:
        # - 分布数が多いNotebookにも360秒の上限を設ける。
        for module in MODULES:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), module], check=True, timeout=360)
