# - 作成日: 2026-09-27
# - 目的: ArviZ 1のPlotCollectionをmarimoで表示するFigureへ変換する。
# - 役割: 旧APIからの移植で繰り返す表示・94% HDI・後始末の不一致を防ぐ。
# - 使用: inference_view(idata, ["mu", "sigma"])。
# - 制約: matplotlibバックエンド、Notebook専用。入力データは変更しない。
# - 非対応: 旧ArviZ・ブラウザー自動操作・EXE配布。
import arviz as az
import marimo as mo
import matplotlib.pyplot as plt


def inference_view(idata, var_names: list[str], *, coords=None):
    """トレース・事後分布・表を表示する。引数はDataTree、変数名、任意の座標。
    戻り値はmarimo表示。前提: posteriorが存在する。副作用は描画のみ。
    不正な座標等は元の例外。例: inference_view(idata, ["p"])。
    """
    trace = az.plot_trace_dist(idata, var_names=var_names, coords=coords,
                               compact=False, backend="matplotlib")
    density = az.plot_dist(idata, var_names=var_names, coords=coords,
                           ci_kind="hdi", ci_prob=0.94, backend="matplotlib")
    summary = az.summary(idata, var_names=var_names, coords=coords,
                         ci_kind="hdi", ci_prob=0.94)
    figures = []
    # - 描画管理から閉じ、再実行時にFigureが蓄積しないようにする。
    for collection in (trace, density):
        figure = collection.get_viz("figure")
        figure.tight_layout()
        plt.close(figure)
        figures.append(figure)
    # - トレース表示だけで発散を見落とさないよう、選択したチェーンの数を併記する。
    if "sample_stats" in idata and "diverging" in idata["sample_stats"]:
        divergences = idata["sample_stats"]["diverging"]
        # - FAQの部分チェーン表示では、全チェーンの発散数を混ぜない。
        if coords and "chain" in coords:
            divergences = divergences.sel(chain=coords["chain"])
        diagnostic = (f"Divergences: {int(divergences.sum())}. "
                      "Inspect R-hat and ESS before interpreting this result.")
    # - 非MCMCの標本を発散0件のMCMC結果と誤解させない。
    else:
        diagnostic = "MCMC divergence statistics are unavailable for these samples."
    return mo.vstack([mo.md(diagnostic), *figures, mo.ui.table(summary.reset_index())])


def model_view(model):
    """モデル図を表示する。引数はPyMCモデル、戻り値はSVG表示。
    前提: Graphvizのdotが使用可能。副作用はdot子プロセスの実行。
    未導入・描画失敗はGraphviz例外。例: model_view(model)。
    """
    import pymc as pm
    return mo.Html(pm.model_to_graphviz(model).pipe(format="svg").decode("utf-8"))
