# - 作成日: 2026-09-27
# - 目的: 第1・3章の確率・正規近似・表示を解析解で検査する。
# - 役割: Notebook専用回帰試験。各Notebookは新規プロセスで起動する。
# - 使用: uv run --locked --group notebook python tests/test_ch01_ch03.py
# - 制約: 固定環境、ローカルキャッシュのみ。ネットワークなし。
# - 非対応: VS Code画面操作・EXE配布。
import subprocess
import sys
from pathlib import Path


# - 直接実行とWindows子プロセスで、移動先から同じNotebookを読み込む。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import numpy as np


def test_chapter(chapter: str) -> None:
    """章を新規状態で検査する。引数は01または03、戻り値なし。
    前提: 固定依存。副作用はセル実行・描画・英語出力。不一致はAssertionError。
    例: test_chapter("01")。キャッシュは登録済みプロジェクト内を使用する。
    """
    print(f"Checking chapter {chapter}", flush=True)
    # - エラーを明示し、元の原因を維持する。
    try:
        # - 第1章は二項分布の解析解と標本の支持を比較する。
        if chapter == "01":
            from ch01_distributions import app
            _, values = app.run()
            np.testing.assert_allclose(values["discrete_probabilities"], np.array([1, 5, 10, 10, 5, 1]) / 32)
            assert abs(values["large_probabilities"].sum() - 1) < 1e-12
            assert values["x_samples"].shape == (1, 500)
            assert set(np.unique(values["x_samples"])) <= set(range(6))
            assert abs(values["x_samples"].mean() - 2.5) < 0.2
            assert abs(values["norm"](0, 0, 1) - 1 / np.sqrt(2 * np.pi)) < 1e-12
        # - 第3章は区間全体の面積と一定密度を確認する。
        else:
            from ch03_bayes import app
            _, values = app.run()
            assert np.all(values["density"] == 1)
            assert np.trapezoid(values["density"], values["coordinates"]) == 1
    # - 失敗を非ゼロ終了として呼出元へ伝える。
    except Exception:
        print(f"Failed: chapter {chapter}\n", flush=True)
        raise
    print(f"Passed: chapter {chapter}\n", flush=True)


# - 各章を同時に計算せず、独立したプロセスで検査する。
if __name__ == "__main__":
    # - 子プロセスは指定された1章だけを実行する。
    if len(sys.argv) == 2:
        test_chapter(sys.argv[1])
    # - 親プロセスは各章に180秒の期限を設ける。
    else:
        # - 前のNotebookの状態を次へ引き継がない。
        for chapter in ("01", "03"):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), chapter], check=True, timeout=180)
