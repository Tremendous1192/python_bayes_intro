# - 作成日: 2026-09-27
# - 目的: 診断表示が発散数とチェーン選択を正しく反映することを検証する。
# - 役割: Notebook専用の表示検証。重いMCMCは再実行しない。
# - 使用: uv run --locked --group notebook python tests/test_notebook_plots.py
# - 制約: 合成DataTree、matplotlibの画面なし描画。
# - 非対応: 統計的収束の保証・VS Code画面操作・EXE。
import sys
from pathlib import Path

# - 直接実行とWindows子プロセスで、移動先から同じNotebookを読み込む。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import xarray as xr

from mod_plots import inference_view


def test_diagnostics_display() -> None:
    """- 全体・部分チェーンの発散表示と入力の保持を検査する。
    - 引数・戻り値: なし。前提: 固定ArviZ環境。副作用: 描画・英語出力。
    - 失敗: AssertionError等を伝える。使用例: test_diagnostics_display()。
    """
    print("Checking diagnostic display and chain selection", flush=True)
    # - 表示API変更によるエラーを失敗として伝える。
    try:
        rng = np.random.default_rng(42)
        divergences = np.zeros((4, 100), dtype=bool)
        divergences[0, 0] = True
        divergences[2, :2] = True
        posterior = xr.Dataset({"mu": (("chain", "draw"), rng.normal(size=(4, 100)))},
                               coords={"chain": np.arange(4), "draw": np.arange(100)})
        stats = xr.Dataset({"diverging": (("chain", "draw"), divergences)},
                           coords={"chain": np.arange(4), "draw": np.arange(100)})
        idata = xr.DataTree.from_dict({"posterior": posterior, "sample_stats": stats})
        full_view = inference_view(idata, ["mu"])
        partial_view = inference_view(idata, ["mu"], coords={"chain": [0, 1]})
        assert "Divergences: 3" in full_view.text
        assert "Divergences: 1" in partial_view.text
        np.testing.assert_array_equal(idata["sample_stats"]["diverging"], divergences)
        assert idata["posterior"].sizes["chain"] == 4
        assert not matplotlib.pyplot.get_fignums()
    # - 原因を保持して失敗を報告する。
    except Exception:
        print("Failed: diagnostic display\n", flush=True)
        raise
    print("Passed: diagnostic display\n", flush=True)


# - 直接実行だけで検査を開始する。
if __name__ == "__main__":
    test_diagnostics_display()
