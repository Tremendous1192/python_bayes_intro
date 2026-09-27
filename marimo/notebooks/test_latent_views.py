# - 作成日: 2026-09-27
# - 目的: 潜在モデルの実際の表示セルで観測と推論の凡例を区別できることを検査する。
# - 役割: Notebook専用の合成事後分布による表示検証。MCMCの再計算は行わない。
# - 使用: uv run --locked --group notebook python notebooks/test_latent_views.py
# - 制約: 合成標本、固定観測、.cache/validation内の画像だけを保存する。
# - 非対応: 合成標本での統計的受入・VS Code画面操作・EXE。
import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import xarray as xr


def test_latent_legends() -> None:
    """- 3教材の表示セルへ合成標本を渡して凡例を検査する。
    - 引数・戻り値: なし。前提: 固定環境とCSV。
    - 副作用: 描画・検査画像・英語出力。一時的にcloseを差替え、必ず戻す。
    - 失敗: 元の例外を伝える。使用例: test_latent_legends()。
    """
    print("Checking latent plots with synthetic posterior samples", flush=True)
    original_close = plt.close
    target = Path(__file__).resolve().parents[1] / ".cache" / "validation"
    target.mkdir(exist_ok=True)
    # - 検証失敗時も描画APIの差替えを必ず復元する。
    try:
        # - モデル結果のセルだけを置換し、実際の下流表示を検証する。
        for module, components, names in [
            ("ch05_04_latent", 2, ["idata1", "idata2", "idata3"]),
            ("sample_ch05_04_simplified", 2, ["idata1", "idata2"]),
            ("sample_three_class", 3, ["idata3", "idata4", "idata5"]),
        ]:
            rng = np.random.default_rng(42)
            shape = (4, 100)
            means = [1.3, 2.0] if components == 2 else [0.25, 1.3, 2.0]
            variables = {
                "mus": (("chain", "draw", "mus_dim_0"), rng.normal(means, 0.04, (*shape, components))),
                "sigmas": (("chain", "draw", "sigmas_dim_0"), rng.uniform(0.18, 0.25, (*shape, components))),
                "s": (("chain", "draw", "s_dim_0"), rng.integers(0, components, (*shape, components * 50))),
            }
            # - 2クラスはBernoulliのスカラー、3クラスは確率ベクトルを使用する。
            if components == 2:
                variables["p"] = (("chain", "draw"), rng.uniform(0.45, 0.55, shape))
            # - ディリクレ標本は各drawで確率の和が1になる。
            else:
                variables["p"] = (("chain", "draw", "p_dim_0"), rng.dirichlet(np.ones(3), shape))
            posterior = xr.Dataset(variables, coords={"chain": range(4), "draw": range(100)})
            stats = xr.Dataset({"diverging": (("chain", "draw"), np.zeros(shape, dtype=bool))})
            idata = xr.DataTree.from_dict({"posterior": posterior, "sample_stats": stats})
            captured = []

            def capture(figure=None) -> None:
                """- 閉じる直前に対象図の凡例を検査する。
                - 引数: Figure等。戻り値: なし。前提: plt.closeの互換呼出。
                - 副作用: 対象画像保存・Figure閉鎖。失敗: 検査例外。
                - 使用例: capture(figure)。
                """
                # - ArviZ等の別の図は変更せず、重ね描きだけを対象にする。
                if isinstance(figure, Figure) and figure.axes and figure.axes[0].get_title() in {"Precision prior", "Ordered means"}:
                    ax = figure.axes[0]
                    species_legends = [artist for artist in ax.artists
                                       if hasattr(artist, "get_title") and artist.get_title().get_text() == "Observed species"]
                    assert len(species_legends) == 1
                    assert len(species_legends[0].get_texts()) == components
                    assert len(ax.get_legend().get_texts()) == components
                    title = ax.get_title().replace(" ", "-")
                    figure.savefig(target / f"synthetic-{module}-{title}.png", dpi=120, bbox_inches="tight")
                    captured.append(title)
                original_close(figure)

            plt.close = capture
            # - 元の推論セルの出力をすべて指定し、MCMCを実行させない。
            importlib.import_module(module).app.run(defs={name: idata for name in names})
            assert len(captured) == (1 if module == "sample_ch05_04_simplified" else 2)
            plt.close = original_close
        assert not plt.get_fignums()
    # - 表示エラーを明示して原因を保持する。
    except Exception:
        print("Failed: latent plot legends\n", flush=True)
        raise
    # - 後続の描画が検査フックを使わないようにする。
    finally:
        plt.close = original_close
    print("Passed: latent plot legends\n", flush=True)


# - 直接起動時だけ合成表示を検証する。
if __name__ == "__main__":
    test_latent_legends()
