# - 作成日: 2026-09-27
# - 目的: 第2章 2.3 正規分布をVS Codeのmarimoで学習する。
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
    """- Iris観測値と理論密度を描く。
    - 引数: mo, np, plt, stats。戻り値: setosa。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    from notebook_data import load_data
    iris = load_data("iris.csv")
    setosa = iris.loc[iris["species"] == "setosa", "sepal_length"]
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].hist(setosa, bins=np.arange(4.0, 6.2, 0.2))
    _axes[0].set(title="Setosa", xlabel="Sepal length")
    _x = np.arange(-8.0, 10.0, 0.01)
    _axes[1].plot(_x, stats.norm.pdf(_x, 3, 2), label="mu=3 sigma=2")
    _axes[1].plot(_x, stats.norm.pdf(_x, 1, 3), label="mu=1 sigma=3")
    _axes[1].set(xlabel="Value", ylabel="Density")
    _axes[1].legend()
    plt.close(_figure)
    mo.vstack([mo.md("## 2.3 正規分布\nIrisの観測分布と、平均・標準偏差による密度の違いを確認します。"), _figure])
    return (setosa,)


@app.cell
def _(pm):
    """- 正規分布2例をサンプリングする。
    - 引数: pm。戻り値: priors。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    priors = {}
    # - 標準正規分布は平均0、標準偏差1。
    with pm.Model() as model4:
        pm.Normal("x", mu=0.0, sigma=1.0)
        priors["Normal mu=0 sigma=1"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 平均と標準偏差を変更した例を独立して作る。
    with pm.Model() as model5:
        pm.Normal("x", mu=3.0, sigma=2.0)
        priors["Normal mu=3 sigma=2"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    return (priors,)


@app.cell
def _(az, mo, plt, priors):
    """- 各分布の事前予測・統計表・密度図を比較する。
    - 引数: az, mo, plt, priors。戻り値: tables。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    tables = {}
    figures = []
    # - 各分布の理論パラメータを維持し、結果を独立して表示する。
    for label, prior in priors.items():
        tables[label] = az.summary(prior, group="prior", kind="stats", ci_kind="hdi", ci_prob=0.94)
        _collection = az.plot_dist(prior, group="prior", backend="matplotlib", ci_kind="hdi", ci_prob=0.94)
        _figure = _collection.get_viz("figure")
        _figure.suptitle(label)
        plt.close(_figure)
        figures.append(mo.vstack([mo.md(label), mo.ui.table(tables[label].reset_index()), _figure]))
    mo.vstack(figures)
    return (tables,)


@app.cell
def _(np, plt, priors):
    """- 元Notebookのヒストグラム表示を保持する。
    - 引数: np, plt, priors。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _histogram, _axes = plt.subplots(1, 2, figsize=(10, 3))
    # - 密度図と同じサンプルを度数ヒストグラムでも確認する。
    for _axis, (_label, _prior) in zip(_axes, priors.items()):
        _axis.hist(_prior["prior"]["x"].values.ravel(), bins=15)
        _axis.set(title=_label, xlabel="Value", ylabel="Count")
    plt.close(_histogram)
    _histogram
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
