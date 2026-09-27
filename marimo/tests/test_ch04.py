# - 作成日: 2026-09-27
# - 目的: 第4章の最尤値・事後分布・反復条件・離散変数を検査する。
# - 役割: Notebook専用の新規プロセス検証。
# - 使用: uv run --locked --group notebook python tests/test_ch04.py
# - 制約: 各ケース1800秒。Book runの結果と短縮検査を区別する。
# - 非対応: VS Code画面操作・EXE配布。
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
from scipy import stats

# - 各ファイルは元Notebookの自然な学習単位に対応する。
MODULES = ("ch04_likelihood", "ch04_posterior", "ch04_prior_comparison", "sample_ch04_figure")


def test_chapter(module: str, mode: str = "book") -> None:
    """章を検査する。引数はモジュール名とbook/quick/off、戻り値なし。
    前提: 固定環境とGraphviz。副作用は計算・描画・英語出力。
    不一致はAssertionError。例: test_chapter("ch04_posterior", "quick")。
    """
    print(f"Checking {module}: {mode}", flush=True)
    # - 数値不一致とセルの例外を失敗として伝える。
    try:
        app = importlib.import_module(module).app
        overrides = {}
        # - 最尤推定以外は実行ボタンとモードを検査用に指定する。
        if module != "ch04_likelihood" and mode != "off":
            overrides = {"run_inference": SimpleNamespace(value=True),
                         "sampling_mode": SimpleNamespace(value=mode)}
        # - 補足のPDFもプロジェクト内の一時出力として検査する。
        if module == "sample_ch04_figure" and mode == "book":
            overrides["export_pdf"] = SimpleNamespace(value=True)
        _, values = app.run(defs=overrides)
        # - 未押下時にMCMC結果を作らないことを確認する。
        if mode == "off":
            assert "idata1_1" not in values and "idata1_2" not in values
        # - 最尤値は解析解2/5と比較する。
        elif module == "ch04_likelihood":
            # - 40反復の解析勾配による参照値。最適値との差は約1.21e-5残る。
            assert abs(values["maximum_likelihood"] - 0.39998790415577956) < 2e-7
            assert abs(values["maximum_likelihood"] - 0.4) < 2e-5
            assert values["optimization_log"][-1, 2] < values["optimization_log"][0, 2]
        # - ベルヌーイと二項分布の両方でBeta(3,4)を検査する。
        elif module == "ch04_posterior":
            # - 乱数列の一致ではなく、独立した解析解の平均で判定する。
            for name in ("idata1_1", "idata1_2", "idata2"):
                result = values[name]["posterior"]["p"]
                assert np.isfinite(result).all()
                assert abs(float(result.mean()) - 3 / 7) < (0.03 if mode == "book" else 0.12)
                # - 本実行では平均だけでなくBeta(3,4)の分散も確認する。
                if mode == "book":
                    assert abs(float(result.var()) - 12 / (49 * 8)) < 0.004
            assert values["idata1_1"]["posterior"].sizes["chain"] == 3
            assert values["idata1_1"]["posterior"].sizes["draw"] == (2000 if mode == "book" else 100)
        # - 通常と区間を制限した事前分布の解析解を検査する。
        elif module == "ch04_prior_comparison":
            assert abs(float(values["idata3"]["posterior"]["p"].mean()) - 21 / 52) < 0.03
            restricted = values["idata4"]["posterior"]["p"].values
            expected = stats.beta.expect(lambda p: p, args=(3, 4), lb=0.1, ub=0.9, conditional=True)
            assert np.all((restricted >= 0.1) & (restricted <= 0.9))
            assert abs(restricted.mean() - expected) < 0.03
        # - 補足モデルは離散予測変数を保持し、値域とチェーン形状を検査する。
        else:
            # - PDFの作成場所と形式を確認し、ブラウザーの保存先には依存しない。
            if mode == "book":
                assert values["pdf_path"].is_relative_to(Path(__file__).resolve().parents[1] / ".cache")
                assert values["pdf_path"].read_bytes().startswith(b"%PDF-")
            # - 混合の悪い例を解析モデルへ置換せず、診断は画面に表示する。
            for name in ("prediction_result", "nested_result"):
                result = values[name]["posterior"]
                y = result["Y_pred"].values
                assert np.all((y >= 0) & (y <= 1000) & (y == np.floor(y)))
                assert np.all((result["p"].values > 0) & (result["p"].values < 1))
    # - 失敗の詳細と非ゼロ終了を維持する。
    except Exception:
        print(f"Failed: {module}, {mode}\n", flush=True)
        raise
    print(f"Passed: {module}, {mode}\n", flush=True)


# - 新規プロセスで各Notebookを検査する。
if __name__ == "__main__":
    # - 子プロセスでは指定された条件だけを処理する。
    if len(sys.argv) >= 2:
        test_chapter(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "book")
    # - 親プロセスは実行停止と各教材の本来の反復数を順番に検査する。
    else:
        cases = [("ch04_posterior", "off")] + [(module, "book") for module in MODULES]
        # - 同時推論を避け、各ケースに30分の期限を設ける。
        for module, mode in cases:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), module, mode],
                           check=True, timeout=1800)
