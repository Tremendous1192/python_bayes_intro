# - 作成日: 2026-09-27
# - 目的: IRTの全観測・名前付き座標・MCMC/ADVI・同点比較を検証する。
# - 役割: Notebook専用の新規プロセス検証。
# - 使用: uv run --locked --group notebook python notebooks/test_ch06_03.py book
# - 制約: 全1000人50問、CPU同時2チェーン、最大7200秒。quickは配線確認専用。
# - 非対応: GUI・GPU・EXE・短縮標本による収束保証。
import multiprocessing
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np

from ch06_03_irt import app
from irt_model import ability_summary


def test_irt_preparation() -> None:
    """- ボタン未押下ではデータ準備だけを実行する。
    - 引数・戻り値: なし。前提: 固定CSV。副作用: 読込・図・英語出力。
    - 失敗: AssertionError等を伝える。使用例: test_irt_preparation()。
    """
    print("Checking IRT preparation and inference gate", flush=True)
    # - 入力と実行開始の制御を長い推論から独立して検証する。
    try:
        _, values = app.run()
        assert values["df"].shape == (1000, 50)
        assert values["response_df"].shape == (50000, 3)
        assert values["model1"]["response_data"].get_value().shape == (50000,)
        assert "idata1" not in values and "idata2" not in values
        np.testing.assert_array_equal(values["df"].to_numpy()[
            values["user_idx"], values["question_idx"]], values["response"])
    # - データ対応やボタン停止の不備を報告する。
    except Exception:
        print("Failed: IRT preparation\n", flush=True)
        raise
    print("Passed: IRT preparation\n", flush=True)


def test_irt_spawn() -> None:
    """- Windowsの2ワーカー起動と全モデルの転送を少数drawで検査する。
    - 引数・戻り値: なし。前提: 固定環境・CSV。副作用: CPU子プロセス・英語出力。
    - 失敗: 起動例外・AssertionError。収束判定には使わない。例: test_irt_spawn()。
    """
    print("Checking IRT Windows workers; not a convergence test", flush=True)
    # - 長い本実行の前に実際の全モデルでspawnと結果の回収を確認する。
    try:
        _, values = app.run()
        result = values["sample_model"](values["model1"], chains=2, draws=5, tune=5,
                                        var_names=["theta", "a", "b"], backend="numba")
        assert result["posterior"]["theta"].shape == (2, 5, 1000)
        assert "logit_p" not in result["posterior"]
    # - 子プロセスの異常を隠さず、本実行前に報告する。
    except Exception:
        print("Failed: IRT Windows workers\n", flush=True)
        raise
    print("Passed: IRT Windows workers\n", flush=True)


def test_irt(mode: str) -> None:
    """- 独立した表の再構成と集計でIRTの結果を検査する。
    - 引数: modeはbookまたはquick。戻り値: なし。前提: 固定データ。
    - 副作用: 長時間推論・描画・英語出力。失敗: 元の例外を伝える。
    - 使用例: test_irt("book")。
    """
    print(f"Checking IRT: {mode}", flush=True)
    # - 推論と表示のどちらの例外も記録して失敗させる。
    try:
        _, values = app.run(defs={"sampling_mode": SimpleNamespace(value=mode),
                                 "run_inference": SimpleNamespace(value=True)})
        df = values["df"]
        assert df.shape == (1000, 50) and values["response"].shape == (50000,)
        restored = values["response_df"].pivot(index="user", columns="question", values="response")
        np.testing.assert_array_equal(restored.loc[df.index, df.columns], df)
        np.testing.assert_array_equal(
            df.to_numpy()[values["user_idx"], values["question_idx"]], values["response"])
        np.testing.assert_allclose(values["f"](np.array([-1, 0, 1])),
                                   [0.26894142137, 0.5, 0.73105857863], atol=1e-11)
        # - 元教材の回数と座標を両方式で保持し、不要な巨大配列は保持しない。
        for name, chains, draws in [("idata1", 4, 1000 if mode == "book" else 100),
                                    ("idata2", 1, 2000 if mode == "book" else 100)]:
            p = values[name]["posterior"]
            assert p["theta"].shape == (chains, draws, 1000)
            assert p["a"].shape == (chains, draws, 50)
            assert (p["a"] > 0).all() and np.isfinite(p["theta"]).all()
            assert "logit_p" not in p
            np.testing.assert_array_equal(p["user"], df.index)
            np.testing.assert_array_equal(p["question"], df.columns)
        assert len(values["mean_field"].hist) == (20000 if mode == "book" else 500)
        assert np.isfinite(values["mean_field"].hist).all()
        scores = values["df_sum1"]
        np.testing.assert_allclose(scores["素点"], df.sum(axis=1) * 2)
        np.testing.assert_allclose(scores[["偏差値", "能力値"]].mean(), 50, atol=1e-10)
        np.testing.assert_allclose(scores[["偏差値", "能力値"]].std(ddof=1), 10, atol=1e-10)
        assert abs(values["df_sum2"]["能力値2"].std(ddof=0) - 10) < 1e-10
        # - 行順を逆転しても受験者名に対応する能力値は変化しない。
        reversed_scores, _, _ = ability_summary(df.iloc[::-1], values["idata1"])
        np.testing.assert_allclose(reversed_scores["能力値"].iloc[::-1], scores["能力値"])
        low, high = values["selected_users"]
        assert scores.loc[low, "素点"] == scores.loc[high, "素点"] == 62
        assert scores.loc[low, "能力値"] <= scores.loc[high, "能力値"]
        difficulty = values["b_mean1"]
        # - 正答した設問だけを抽出する独立した計算で平均困難度を照合する。
        for user in (low, high):
            expected = difficulty[df.loc[user].to_numpy(dtype=bool)].mean()
            assert abs(values["w3"].loc[user] - expected) < 1e-12
        divergences = int(values["idata1"]["sample_stats"]["diverging"].sum())
        print(f"MCMC divergences: {divergences}; selected users: {low}, {high}", flush=True)
        assert not matplotlib.pyplot.get_fignums()
    # - スタックトレースを保持し、失敗を明示する。
    except Exception:
        print(f"Failed: IRT {mode}\n", flush=True)
        raise
    print(f"Passed: IRT {mode}\n", flush=True)


# - 起動時だけ検証を実行し、長時間実行に上限を設ける。
if __name__ == "__main__":
    # - IRTのCPUワーカーがテスト本体を重複起動しないようにする。
    multiprocessing.freeze_support()
    # - 子プロセスでは指定モードの検証を実行する。
    if len(sys.argv) == 3 and sys.argv[2] == "child":
        # - offは全データの準備と未押下の停止だけを検査する。
        if sys.argv[1] == "off":
            test_irt_preparation()
        # - 全モデルを使った少数drawでWindowsのワーカー起動を検査する。
        elif sys.argv[1] == "spawn":
            test_irt_spawn()
        # - quickとbookは実際の推論も検査する。
        else:
            test_irt(sys.argv[1])
    # - 親プロセスはモードを渡し、最大2時間で終了させる。
    else:
        mode = sys.argv[1] if len(sys.argv) > 1 else "book"
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), mode, "child"],
                       check=True, timeout=7200)
