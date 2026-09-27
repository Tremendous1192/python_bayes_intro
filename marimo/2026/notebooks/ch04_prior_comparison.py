# - 作成日: 2026-09-27
# - 目的: 第4章 4.7〜4.9・ArviZ FAQをVS Codeのmarimoで学習する。
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
    from notebook_sampling import sample_model, posterior_mean
    from notebook_plots import inference_view, model_view
    return (mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view,)


@app.cell
def _(mo, pm, model_view):
    """- 標本数・事前分布の変更をモデルで示す。
    - 引数: mo, pm, model_view。戻り値: model3, model4。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("## 4.7〜4.9\n試行回数を50に増やす例と、事前分布を[0.1,0.9]に制限する例を比較します。制限したモデルはBeta(3,4)を同区間で切断した事後分布です。")
    # - 50回中20回成功のモデルはBeta(21,31)と比較できる。
    with pm.Model() as model3:
        _p = pm.Uniform("p", lower=0.0, upper=1.0)
        pm.Binomial("X_obs", p=_p, n=50, observed=20)
    # - 観測5回中2成功を保持し、事前分布だけを変更する。
    with pm.Model() as model4:
        _p = pm.Uniform("p", lower=0.1, upper=0.9)
        pm.Binomial("X_obs", p=_p, n=5, observed=2)
    mo.vstack([model_view(model3), model_view(model4)])
    return (model3, model4,)


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
def _(mo, run_inference, sampling_mode, sample_model, model3, model4):
    """- 入力条件を固定して2モデルを推論する。
    - 引数: mo, run_inference, sampling_mode, sample_model, model3, model4。戻り値: idata3, idata4。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Click Run inference."))
    idata3 = sample_model(model3, sampling_mode.value)
    idata4 = sample_model(model4, sampling_mode.value)
    return (idata3, idata4,)


@app.cell
def _(mo, np, plt, stats, inference_view, idata3, idata4):
    """- 解析的な密度との重ね描きとArviZの軸変更を示す。
    - 引数: mo, np, plt, stats, inference_view, idata3, idata4。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import arviz as az
    _collection = az.plot_dist(idata3, var_names=["p"], backend="matplotlib", ci_kind="hdi", ci_prob=0.94)
    _figure = _collection.get_viz("figure")
    _axis = _figure.axes[0]
    _x = np.linspace(0.001, 0.999, 300)
    _axis.plot(_x, stats.beta.pdf(_x, 21, 31), color="orange", label="Beta(21,31)")
    # - FAQの軸表示とタイトル変更を現行Figureに対して行う。
    _axis.spines["left"].set_visible(True)
    _axis.set(xlabel="p", ylabel="Density", title="Posterior and analytical density")
    _axis.legend()
    plt.close(_figure)
    mo.vstack([inference_view(idata3, ["p"]), _figure, inference_view(idata4, ["p"])])
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
