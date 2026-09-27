# - 作成日: 2026-09-27
# - 目的: ABテストの効果検証をVS Codeのmarimoで学習する。
# - 役割: notebooks/6_1_ABテスト効果検証.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 固定環境の推論・描画を準備する。
    - 引数: なし。戻り値: mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import pymc as pm
    from scipy import stats
    from mod_sampling import sample_model, posterior_mean
    from mod_plots import inference_view, model_view
    return (mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view,)


@app.cell
def _(mo):
    """- 差の符号と比較する2つのデータ量を示す。
    - 引数: mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("""# 6.1 Bayesian A/B tests
    The difference is **B − A**. A negative difference therefore means A has a higher rate.
    Compare small samples (2/40 vs 2/25) with large samples (60/1200 vs 110/1600).
    Uniform priors also give exact Beta posteriors, providing an independent comparison.
    """)
    return


@app.cell
def _(mo):
    """- 反復条件と推論開始を選択する。
    - 引数: mo。戻り値: sampling_mode, run_inference。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 値を読むセルを分け、入力変更だけでは重い計算を始めない。
    sampling_mode = mo.ui.dropdown({"Book run": "book", "Quick check": "quick"}, value="Book run", label="Sampling mode")
    run_inference = mo.ui.run_button(label="Run inference")
    mo.hstack([sampling_mode, run_inference])
    return (sampling_mode, run_inference,)


@app.cell
def _(pm, mo, model_view):
    """- 元教材の二項モデルを定義する。
    - 引数: pm, mo, model_view。戻り値: model_s, model_y。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 小標本のクリック率に独立した一様事前分布を置く。
    with pm.Model() as model_s:
        p_s_a = pm.Uniform("p_s_a", lower=0.0, upper=1.0)
        p_s_b = pm.Uniform("p_s_b", lower=0.0, upper=1.0)
        pm.Binomial("obs_s_a", p=p_s_a, n=40, observed=2)
        pm.Binomial("obs_s_b", p=p_s_b, n=25, observed=2)
        pm.Deterministic("delta_prob_s", p_s_b - p_s_a)
    # - 大標本でも同じ事前分布と差の向きを保持する。
    with pm.Model() as model_y:
        p_y_a = pm.Uniform("p_y_a", lower=0.0, upper=1.0)
        p_y_b = pm.Uniform("p_y_b", lower=0.0, upper=1.0)
        pm.Binomial("obs_y_a", p=p_y_a, n=1200, observed=60)
        pm.Binomial("obs_y_b", p=p_y_b, n=1600, observed=110)
        pm.Deterministic("delta_prob_y", p_y_b - p_y_a)
    mo.vstack([model_view(model_s), model_view(model_y)])
    return (model_s, model_y,)


@app.cell
def _(model_s, model_y, sample_model, sampling_mode, run_inference, mo):
    """- 4チェーンのMCMCを逐次実行する。
    - 引数: model_s, model_y, sample_model, sampling_mode, run_inference, mo。戻り値: idata_s, idata_y。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Press Run inference."))
    idata_s = sample_model(model_s, sampling_mode.value, target_accept=0.99)
    idata_y = sample_model(model_y, sampling_mode.value, target_accept=0.99)
    return (idata_s, idata_y,)


@app.cell
def _(idata_s, idata_y, inference_view, mo):
    """- MCMCの事後分布と診断表を表示する。
    - 引数: idata_s, idata_y, inference_view, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.vstack([inference_view(idata_s, ["p_s_a", "p_s_b", "delta_prob_s"]),
               inference_view(idata_y, ["p_y_a", "p_y_b", "delta_prob_y"])])
    return


@app.cell
def _(pm, run_inference, mo):
    """- 別解の10000回標本化を現在のdraws APIで実行する。
    - 引数: pm, run_inference, mo。戻り値: model_s2, model_y2, samples_s2, samples_y2, delta_a_b_s2, delta_a_b_y2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Press Run inference."))
    # - 成功数+1、失敗数+1から小標本の解析的事後分布を直接標本化する。
    with pm.Model() as model_s2:
        pm.Beta("p_a", alpha=3, beta=39)
        pm.Beta("p_b", alpha=3, beta=24)
        samples_s2 = pm.sample_prior_predictive(draws=10000, random_seed=42)
    # - 大標本もMCMCと独立したベータ標本で比較する。
    with pm.Model() as model_y2:
        pm.Beta("p_a", alpha=61, beta=1141)
        pm.Beta("p_b", alpha=111, beta=1491)
        samples_y2 = pm.sample_prior_predictive(draws=10000, random_seed=42)
    delta_a_b_s2 = (samples_s2["prior"]["p_b"] - samples_s2["prior"]["p_a"]).values.reshape(-1)
    delta_a_b_y2 = (samples_y2["prior"]["p_b"] - samples_y2["prior"]["p_a"]).values.reshape(-1)
    return (model_s2, model_y2, samples_s2, samples_y2, delta_a_b_s2, delta_a_b_y2,)


@app.cell
def _(idata_s, idata_y, delta_a_b_s2, delta_a_b_y2, np, stats, plt, mo):
    """- MCMCと別解の確率を同じ符号で比較する。
    - 引数: idata_s, idata_y, delta_a_b_s2, delta_a_b_y2, np, stats, plt, mo。戻り値: delta_prob_s_values, delta_prob_y_values, n1_rate_s, n1_rate_y, n1_rate_s2, n1_rate_y2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    delta_prob_s_values = idata_s["posterior"]["delta_prob_s"].values.reshape(-1)
    delta_prob_y_values = idata_y["posterior"]["delta_prob_y"].values.reshape(-1)
    n1_rate_s = float(np.mean(delta_prob_s_values < 0))
    n1_rate_y = float(np.mean(delta_prob_y_values < 0))
    n1_rate_s2 = float(np.mean(delta_a_b_s2 < 0))
    n1_rate_y2 = float(np.mean(delta_a_b_y2 < 0))
    _results = [("Small / MCMC", delta_prob_s_values, n1_rate_s),
                ("Large / MCMC", delta_prob_y_values, n1_rate_y),
                ("Small / Beta", delta_a_b_s2, n1_rate_s2),
                ("Large / Beta", delta_a_b_y2, n1_rate_y2)]
    _comparison, _axes = plt.subplots(2, 2, figsize=(11, 7))
    # - カーネル密度の負の領域を塗り、表示した確率の意味を確認する。
    for _ax, (_label, _samples, _rate) in zip(_axes.flat, _results):
        _grid = np.linspace(min(_samples.min(), 0), max(_samples.max(), 0), 300)
        _density = stats.gaussian_kde(_samples)(_grid)
        _ax.plot(_grid, _density)
        _ax.fill_between(_grid, _density, where=_grid < 0, alpha=0.4)
        _ax.axvline(0, color="black", linewidth=1)
        _ax.set(title=f"{_label}: P(A > B) = {_rate:.2%}",
                xlabel="Rate B - rate A", ylabel="Density")
    _comparison.tight_layout()
    plt.close(_comparison)
    _comparison
    return (delta_prob_s_values, delta_prob_y_values, n1_rate_s, n1_rate_y, n1_rate_s2, n1_rate_y2,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
