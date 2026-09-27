# - 作成日: 2026-09-27
# - 目的: 3クラスとFAQのカテゴリ・順序・チェーン選択を検査する。
# - 役割: Notebook専用の本実行検証。
# - 使用: uv run --locked --group notebook python tests/test_latent_references.py
# - 制約: 固定データ、新規プロセス、同時推論1件、各教材3600秒。
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


def test_reference(module: str) -> None:
    """- 支持と順序の制約を検査する。
    - 引数: moduleは教材名。戻り値: なし。前提: データ準備済み。
    - 副作用: 推論・描画・英語出力。失敗: 検査例外を伝える。
    - 使用例: test_reference("sample_three_class")。
    """
    print(f"Checking {module}: book", flush=True)
    # - 推論・検査の失敗を区別せず詳細とともに報告する。
    try:
        _, values = importlib.import_module(module).app.run(defs={
            "sampling_mode": SimpleNamespace(value="book"),
            "run_inference": SimpleNamespace(value=True),
        })
        # - 3クラス教材では事前分布と3つの事後分布を検査する。
        if module == "sample_three_class":
            counts = np.bincount(values["x_samples1"], minlength=3) / 500
            np.testing.assert_allclose(counts, [0.2, 0.5, 0.3], atol=0.08)
            np.testing.assert_allclose(values["x_samples2"].sum(axis=1), 1)
            assert (values["x_samples2"] > 0).all()
            # - 単鎖例と順序制約の4チェーン例で反復条件を変えない。
            for name in ("idata3", "idata4", "idata5"):
                result = values[name]
                posterior = result["posterior"]
                np.testing.assert_allclose(posterior["p"].values.sum(axis=-1), 1, atol=1e-12)
                assert (posterior["p"] > 0).all() and (posterior["sigmas"] > 0).all()
                assert set(np.unique(posterior["s"])).issubset({0, 1, 2})
                assert posterior["s"].shape[-1] == 150
                assert posterior.sizes["chain"] == (4 if name == "idata5" else 1)
                assert posterior.sizes["draw"] == (1000 if name == "idata5" else 2000)
                print(f"{name}: divergences={int(result['sample_stats']['diverging'].sum())}", flush=True)
            assert (np.diff(values["idata5"]["posterior"]["mus"].values, axis=-1) > 0).all()
            reference = values["idata3"]["posterior"]
            predicted = (reference["p"].values * reference["mus"].values).sum(axis=-1).mean()
            assert abs(predicted - values["X"].mean()) < 0.2
        # - FAQの選択は実際に存在する5チェーンを重複なくすべて覆う。
        else:
            result = values["idata1_2"]
            posterior = result["posterior"]
            assert posterior["s"].shape == (5, 1000, 100)
            assert values["chain_groups"] == [[0, 1], [2], [3, 4]]
            np.testing.assert_array_equal(posterior["chain"], np.arange(5))
            assert set(np.unique(posterior["s"])).issubset({0, 1})
            print(f"FAQ: divergences={int(result['sample_stats']['diverging'].sum())}", flush=True)
        assert not matplotlib.pyplot.get_fignums()
    # - 失敗の根拠を失わず終了する。
    except Exception:
        print(f"Failed: {module}\n", flush=True)
        raise
    print(f"Passed: {module}\n", flush=True)


# - 各教材を新規プロセスで順番に検査する。
if __name__ == "__main__":
    # - 子プロセスの対象を1教材に限定する。
    if len(sys.argv) == 2:
        test_reference(sys.argv[1])
    # - 親では各実行を最大1時間に制限する。
    else:
        # - 長い推論同士のCPU競合を避ける。
        for module in ("sample_three_class", "sample_latent_faq"):
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), module],
                           check=True, timeout=3600)
