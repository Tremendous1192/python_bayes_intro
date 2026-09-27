# - 作成日: 2026-09-27
# - 目的: 第4章 4.3〜4.6 事後分布をVS Codeのmarimoで学習する。
# - 役割: notebooks/4章_はじめてのベイズ推論実習.ipynbの対応する学習内容を移植した原本。
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
def _(mo, np, pm, model_view):
    """- 2通りの尤度とモデル構造を確認する。
    - 引数: mo, np, pm, model_view。戻り値: model1, model2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _introduction = mo.md('## 4.3 to 4.6\nCompare five Bernoulli observations with two binomial successes. Both posteriors are Beta(3,4). Book run uses 3 chains of 2,000 draws for the explicit example and 4 chains of 1,000 draws for the standard example.')
    X = np.array([1, 0, 0, 1, 0])
    # - 個々の観測値をベルヌーイ分布へ渡す。
    with pm.Model() as model1:
        _p = pm.Uniform("p", lower=0.0, upper=1.0)
        pm.Bernoulli("X_obs", p=_p, observed=X)
    # - 同じ観測を成功数に集約したモデルを作る。
    with pm.Model() as model2:
        _p = pm.Uniform("p", lower=0.0, upper=1.0)
        pm.Binomial("X_obs", p=_p, n=5, observed=2)
    mo.vstack([_introduction, model_view(model1), model_view(model2)])
    return (model1, model2,)


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
def _(mo, run_inference, sampling_mode, sample_model, model1, model2):
    """- 明示条件・標準条件・二項分布を別々に推論する。
    - 引数: mo, run_inference, sampling_mode, sample_model, model1, model2。戻り値: idata1_1, idata1_2, idata2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Click Run inference."))
    idata1_1 = sample_model(model1, sampling_mode.value, chains=3, draws=2000, tune=2000)
    idata1_2 = sample_model(model1, sampling_mode.value)
    idata2 = sample_model(model2, sampling_mode.value)
    return (idata1_1, idata1_2, idata2,)


@app.cell
def _(mo, inference_view, idata1_1, idata1_2, idata2):
    """- トレース・94% HDI・統計表で比較する。
    - 引数: mo, inference_view, idata1_1, idata1_2, idata2。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.vstack([mo.md('### Explicit sampling settings'), inference_view(idata1_1, ["p"]),
               mo.md('### Standard settings'), inference_view(idata1_2, ["p"]),
               mo.md('### The same problem with a binomial likelihood'), inference_view(idata2, ["p"])])
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
