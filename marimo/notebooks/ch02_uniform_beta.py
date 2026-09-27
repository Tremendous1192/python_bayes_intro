# - 作成日: 2026-09-27
# - 目的: 第2章 2.4〜2.6 連続分布をVS Codeのmarimoで学習する。
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
def _(np):
    """- ベータ密度の正規化を正しい式で実装する。
    - 引数: np。戻り値: Beta。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    from scipy.special import betaln

    def Beta(p, alpha: float, beta: float):
        """ベータ密度を返す。引数は0<p<1の座標、正のalpha・beta。戻り値は密度。
        前提: 内点の計算。副作用なし。型不正は計算例外。例: Beta(0.5, 2, 2)。
        旧式はalpha=3,beta=4で偶然一致したため、一般の正規化係数に修正する。
        """
        return np.exp((alpha - 1) * np.log(p) + (beta - 1) * np.log1p(-p) - betaln(alpha, beta))
    return (Beta,)


@app.cell
def _(Beta, mo, np, plt):
    """- ベータ密度の数式と図を対応させる。
    - 引数: Beta, mo, np, plt。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _p = np.linspace(0.001, 0.999, 300)
    _beta_figure, _axis = plt.subplots(figsize=(6, 3))
    _axis.plot(_p, Beta(_p, 3, 4))
    _axis.set(title="Beta alpha=3 beta=4", xlabel="p", ylabel="Density")
    plt.close(_beta_figure)
    mo.vstack([mo.md('## 2.4 to 2.6\nCompare uniform, beta, and half-normal distributions. The Beta density integrates to 1 over its support.'), _beta_figure])
    return


@app.cell
def _(pm):
    """- 連続分布5例のパラメータとサンプルを比較する。
    - 引数: pm。戻り値: priors。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    priors = {}
    # - 区間[0,1]の一様分布を作る。
    with pm.Model() as model6:
        pm.Uniform("x", lower=0.0, upper=1.0)
        priors["Uniform 0 to 1"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 区間だけを変更した一様分布を作る。
    with pm.Model() as model7:
        pm.Uniform("x", lower=0.1, upper=0.9)
        priors["Uniform 0.1 to 0.9"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 成功2回・失敗3回に対応するベータ分布を作る。
    with pm.Model() as model8:
        pm.Beta("p", alpha=3, beta=4)
        priors["Beta 3,4"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 成功20回・失敗30回に増やした場合の集中を確認する。
    with pm.Model() as model9:
        pm.Beta("p", alpha=21, beta=31)
        priors["Beta 21,31"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 正の標準偏差などの事前分布に使う半正規分布を作る。
    with pm.Model() as model10:
        pm.HalfNormal("x", sigma=1.0)
        priors["HalfNormal sigma=1"] = pm.sample_prior_predictive(draws=500, random_seed=42)
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
def _(plt, priors, mo):
    """- 度数ヒストグラムを併記する。
    - 引数: plt, priors, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _histograms = []
    # - 同じサンプルの度数表示を併記して密度との単位の違いを示す。
    for _label, _prior in priors.items():
        _group = _prior["prior"].dataset
        _values = next(iter(_group.data_vars.values())).values.ravel()
        _figure, _axis = plt.subplots(figsize=(5, 2))
        _axis.hist(_values, bins=15)
        _axis.set(title=_label, xlabel="Value", ylabel="Count")
        plt.close(_figure)
        _histograms.append(_figure)
    mo.vstack(_histograms)
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
