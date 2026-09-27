# - 作成日: 2026-09-27
# - 目的: 第1章 確率分布をVS Codeのmarimoで学習する。
# - 役割: notebooks/1章_確率分布とは.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 計算・表・図のライブラリを準備する。
    - 引数: なし。戻り値: mo, np, pd, plt, pm, az, stats。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pymc as pm
    import arviz as az
    from scipy import stats
    mo.md("## 第1章 確率分布\n二項分布、正規近似、事前予測サンプルを比較します。")
    return (mo, np, pd, plt, pm, az, stats,)


@app.cell
def _(np, plt, stats):
    """- 離散分布と試行数の増加を可視化する。
    - 引数: np, plt, stats。戻り値: discrete_probabilities, large_probabilities。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 二項確率は大きな組合せ数を直接生成せず、同じ式を安定に評価する。
    discrete_probabilities = stats.binom.pmf(np.arange(6), 5, 0.5)
    large_x = np.arange(1001)
    large_probabilities = stats.binom.pmf(large_x, 1000, 0.5)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].bar(np.arange(6), discrete_probabilities)
    axes[0].set(title="Binomial: n=5", xlabel="Successes", ylabel="Probability")
    axes[1].bar(large_x, large_probabilities)
    axes[1].set(xlim=(430, 570), title="Binomial: n=1000", xlabel="Successes")
    figure.tight_layout()
    plt.close(figure)
    figure
    return (discrete_probabilities, large_probabilities,)


@app.cell
def _(np):
    """- 書籍の正規密度の数式を定義する。
    - 引数: np。戻り値: norm。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    def norm(x, mu: float, sigma: float):
        """正規密度を計算する。引数は座標・平均・正の標準偏差、戻り値は密度。
        前提: sigma>0。副作用なし。型不正時は計算例外。例: norm(0, 0, 1)。
        """
        return np.exp(-((x - mu) / sigma) ** 2 / 2) / (np.sqrt(2 * np.pi) * sigma)
    return (norm,)


@app.cell
def _(norm, np, plt, stats):
    """- 正規近似と区間確率の面積を比較する。
    - 引数: norm, np, plt, stats。戻り値: normal_density, interval_probability。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - n=1000、p=0.5の平均500・標準偏差sqrt(250)で正規近似する。
    normal_x = np.arange(430, 571)
    normal_density = norm(normal_x, 500, np.sqrt(250))
    interval_probability = stats.norm.cdf(480, 500, np.sqrt(250)) - stats.norm.cdf(460, 500, np.sqrt(250))
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].bar(normal_x, stats.binom.pmf(normal_x, 1000, 0.5), label="Binomial")
    _axes[0].plot(normal_x, normal_density, color="black", label="Normal")
    _axes[0].legend()
    _axes[1].plot(normal_x, normal_density)
    _axes[1].fill_between(normal_x, 0, normal_density, where=(normal_x >= 460) & (normal_x <= 480))
    _axes[1].set(title=f"Area from 460 to 480: {interval_probability:.4f}", xlabel="Successes")
    _fig.tight_layout()
    plt.close(_fig)
    _fig
    return (normal_density, interval_probability,)


@app.cell
def _(pm):
    """- 固定シードで事前予測を生成する。
    - 引数: pm。戻り値: prior_samples。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    model = pm.Model()
    # - 観測値を与えず、確率変数xの事前予測を生成する。
    with model:
        x = pm.Binomial("x", p=0.5, n=5)
        prior_samples = pm.sample_prior_predictive(draws=500, random_seed=42)
    return (prior_samples,)


@app.cell
def _(az, mo, pd, plt, prior_samples):
    """- DataTree・頻度表・統計量・ArviZの図を表示する。
    - 引数: az, mo, pd, plt, prior_samples。戻り値: x_samples, summary。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    x_samples = prior_samples["prior"]["x"].values
    value_counts = pd.Series(x_samples.reshape(-1), name="Successes").value_counts().sort_index()
    summary = az.summary(prior_samples, group="prior", kind="stats", ci_kind="hdi", ci_prob=0.94)
    # - ArviZ 1の戻り値からFigureを取り出す。Axesを返す旧APIとは異なる。
    _collection = az.plot_dist(prior_samples, group="prior", backend="matplotlib", ci_kind="hdi", ci_prob=0.94)
    _figure = _collection.get_viz("figure")
    plt.close(_figure)
    mo.vstack([mo.md("### サンプル値の確認\nDataTreeのpriorグループから値を取り出します。"),
               prior_samples, mo.ui.table(value_counts.reset_index()), mo.ui.table(summary.reset_index()), _figure])
    return (x_samples, summary,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
