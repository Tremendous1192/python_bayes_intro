# - 作成日: 2026-09-27
# - 目的: 2クラス潜在モデルの観測対応・支持・制約を検査する。
# - 役割: 本編と簡略版のNotebook専用検証。
# - 使用: uv run --locked --group notebook python tests/test_ch05_04.py
# - 制約: 本実行を1件ずつ、新規プロセスで最大3600秒実行する。
# - 非対応: 単鎖の収束保証・GUI・EXE。
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


def test_latent(module: str) -> None:
    """- 潜在モデルを独立した支持・モーメント条件で検査する。
    - 引数: moduleはNotebook名。戻り値: なし。前提: 固定データ準備済み。
    - 副作用: 推論・描画・英語出力。失敗: AssertionError等を伝える。
    - 使用例: test_latent("ch05_04_latent")。
    """
    print(f"Checking {module}: book", flush=True)
    # - 推論例外をテスト失敗として明示する。
    try:
        _, values = importlib.import_module(module).app.run(defs={
            "sampling_mode": SimpleNamespace(value="book"),
            "run_inference": SimpleNamespace(value=True),
        })
        np.testing.assert_array_equal(values["indexes"], [7, 1, 27, 60, 50])
        assert values["X"].shape == (100,) and values["sval"].shape == (5, 2000)
        names = ["idata1", "idata2"]
        # - 本編にだけ平均順序を制約した第3モデルが存在する。
        if module == "ch05_04_latent":
            names.append("idata3")
            ordered = values["idata3"]["posterior"]["mus"].values
            assert (ordered[..., 1] > ordered[..., 0]).all()
        # - 簡略版では混合確率をサンプリングしないことを確認する。
        else:
            assert "p" not in values["model1"].named_vars
            assert "p" not in values["model2"].named_vars
        # - 比較例も支持を満たす必要があるが、収束は別途診断表で判断する。
        for name in names:
            result = values[name]
            posterior = result["posterior"]
            assert set(np.unique(posterior["s"])).issubset({0, 1})
            assert np.isfinite(posterior["mus"]).all()
            assert (posterior["sigmas"] > 0).all()
            assert posterior.sizes["chain"] == (4 if name == "idata3" else 1)
            assert posterior.sizes["draw"] == (2000 if name == "idata1" else 1000)
            print(f"{name}: divergences={int(result['sample_stats']['diverging'].sum())}", flush=True)
        # - 基準モデルの混合平均はラベルの交換によらず観測平均と一致する。
        reference = values["idata1"]["posterior"]
        means = reference["mus"].values
        probability = reference["p"].values if "p" in reference else 0.5
        mixture_mean = ((1 - probability) * means[..., 0] + probability * means[..., 1]).mean()
        assert abs(mixture_mean - values["X"].mean()) < 0.2
        assert not matplotlib.pyplot.get_fignums()
    # - 例外の原因を保持したまま失敗を出力する。
    except Exception:
        print(f"Failed: {module}\n", flush=True)
        raise
    print(f"Passed: {module}\n", flush=True)


# - 前のNotebook状態を持ち越さず、CPU推論は同時に1件だけ実施する。
if __name__ == "__main__":
    # - 子プロセスは指定された教材を検査する。
    if len(sys.argv) == 2:
        test_latent(sys.argv[1])
    # - 親プロセスは順序を守り、各教材の実行時間を制限する。
    else:
        # - 本編と簡略版を別プロセスで評価する。
        for module in ("ch05_04_latent", "sample_ch05_04_simplified"):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), module],
                           check=True, timeout=3600)
