# - 作成日: 2026-09-27
# - 目的: 第2章 2.1・2.2 離散分布をVS Codeのmarimoで学習する。
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
def _(mo, pm):
    """- 離散分布3例をサンプリングする。
    - 引数: mo, pm。戻り値: priors。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("## 2.1 ベルヌーイ分布・2.2 二項分布\n成功確率0.5を固定し、試行回数を1、5、50と変えます。")
    priors = {}
    # - 成功・失敗の1回試行をモデル化する。
    with pm.Model() as model1:
        pm.Bernoulli("x", p=0.5)
        priors["Bernoulli p=0.5"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 5回中の成功数の確率分布を定義する。
    with pm.Model() as model2:
        pm.Binomial("x", p=0.5, n=5)
        priors["Binomial n=5"] = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 50回に増やし、平均と分散の変化を確認する。
    with pm.Model() as model3:
        pm.Binomial("x", p=0.5, n=50)
        priors["Binomial n=50"] = pm.sample_prior_predictive(draws=500, random_seed=42)
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


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
