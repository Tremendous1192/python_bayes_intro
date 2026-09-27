# - 作成日: 2026-09-27
# - 目的: 第2章 CIとHDIの違いをVS Codeのmarimoで学習する。
# - 役割: notebooks/2章_よく利用される確率分布.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 確率分布のライブラリを準備する。
    - 引数: なし。戻り値: mo, np, plt, pm, az, stats。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pymc as pm
    import arviz as az
    from scipy import stats
    return (mo, np, plt, pm, az, stats,)


@app.cell
def _(mo, np, plt, stats):
    """- 端点・確率質量・密度しきい値を確認する。
    - 引数: mo, np, plt, stats。戻り値: ci_bounds, ci_mass, hdi_bounds, hdi_mass。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    from scipy.optimize import brentq
    # - 自由度3のカイ二乗分布。元のCI例と同じ中央80%を用いる。
    ci_bounds = stats.chi2.ppf([0.1, 0.9], df=3)
    ci_mass = stats.chi2.cdf(ci_bounds[1], 3) - stats.chi2.cdf(ci_bounds[0], 3)
    # - 元Notebookの密度しきい値0.05を保ち、端点は数値的に求める。
    hdi_bounds = np.array([brentq(lambda x: stats.chi2.pdf(x, 3) - 0.05, 1e-12, 1),
                           brentq(lambda x: stats.chi2.pdf(x, 3) - 0.05, 1, 30)])
    hdi_mass = stats.chi2.cdf(hdi_bounds[1], 3) - stats.chi2.cdf(hdi_bounds[0], 3)
    _x = np.linspace(0, 15, 500)
    _y = stats.chi2.pdf(_x, 3)
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3))
    # - この教材の2区間は同じ確率質量ではない点も明記する。
    for _axis, _bounds, _title in zip(_axes, [ci_bounds, hdi_bounds], ["Central interval", "Density threshold 0.05"]):
        _axis.plot(_x, _y)
        _axis.fill_between(_x, 0, _y, where=(_x >= _bounds[0]) & (_x <= _bounds[1]))
        _axis.set(title=_title, xlabel="Value", ylabel="Density")
    _axes[1].axhline(0.05, color="black", linestyle="--")
    plt.close(_figure)
    mo.vstack([mo.md("## CIとHDIの違い\n元の例は中央80%区間と密度0.05以上の区間です。両者の確率質量は同一ではありません。"),
               mo.ui.table({"Interval": ["Central", "Density threshold"], "Mass": [ci_mass, hdi_mass]}), _figure])
    return (ci_bounds, ci_mass, hdi_bounds, hdi_mass,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
