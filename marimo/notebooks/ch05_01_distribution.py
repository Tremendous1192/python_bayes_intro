# - 作成日: 2026-09-27
# - 目的: 5.1 データ分布のベイズ推論をVS Codeのmarimoで学習する。
# - 役割: notebooks/5_1_データ分布のベイズ推論.ipynbの対応する学習内容を移植した原本。
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
def _(mo, np, plt):
    """- 固定Irisデータを確認し、比較用の標本を用意する。
    - 引数: mo, np, plt。戻り値: df1, X, X_less, sns。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    from notebook_data import load_data
    import seaborn as sns
    df = load_data("iris.csv")
    df1 = df.query('species == "setosa"')
    X = df1["sepal_length"].to_numpy()
    X_less = X[:5]
    _figure, _axis = plt.subplots(figsize=(6, 3))
    sns.histplot(data=df1, x="sepal_length", bins=np.arange(4.0, 6.2, 0.2), kde=True, ax=_axis)
    _axis.set(title="Setosa sepal length", xlabel="Sepal length", ylabel="Count")
    plt.close(_figure)
    mo.vstack([mo.md("## 5.1 データ分布の推定\n50件と先頭5件の推論を比較します。精度tauを使う例はsigmaへの事前分布も変わるため、単なる変数名変更ではありません。"),
               mo.ui.table(df.iloc[[0, 1, 50, 51, 100, 101]]), mo.ui.table(df.head()), _figure])
    return (df1, X, X_less, sns,)


@app.cell
def _(pm, X, X_less, mo, model_view):
    """- 全標本・少数標本・精度パラメータのモデルを定義する。
    - 引数: pm, X, X_less, mo, model_view。戻り値: model1, model2, model3。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 全50件を平均mu・標準偏差sigmaでモデル化する。
    with pm.Model() as model1:
        _mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        _sigma = pm.HalfNormal("sigma", sigma=10.0)
        pm.Normal("X_obs", mu=_mu, sigma=_sigma, observed=X)
    # - 同じ事前分布で観測を先頭5件に絞る。
    with pm.Model() as model2:
        _mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        _sigma = pm.HalfNormal("sigma", sigma=10.0)
        pm.Normal("X_obs", mu=_mu, sigma=_sigma, observed=X_less)
    # - 精度tau=1/sigma²への事前分布を使うコラムを保持する。
    with pm.Model() as model3:
        _mu = pm.Normal("mu", mu=0.0, sigma=10.0)
        _tau = pm.HalfNormal("tau", sigma=10.0)
        pm.Normal("X_obs", mu=_mu, tau=_tau, observed=X)
        pm.Deterministic("sigma", 1 / pm.math.sqrt(_tau))
    mo.vstack([model_view(model1), model_view(model3)])
    return (model1, model2, model3,)


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
def _(mo, run_inference, sampling_mode, sample_model, model1, model2, model3):
    """- 同じシードで3モデルを個別に推論する。
    - 引数: mo, run_inference, sampling_mode, sample_model, model1, model2, model3。戻り値: idata1, idata2, idata3。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Click Run inference."))
    idata1 = sample_model(model1, sampling_mode.value)
    # - 現行環境の既定0.8で発散1回を検出したため、小標本だけ受容率を0.95にする。
    idata2 = sample_model(model2, sampling_mode.value, target_accept=0.95)
    idata3 = sample_model(model3, sampling_mode.value)
    return (idata1, idata2, idata3,)


@app.cell
def _(mo, inference_view, idata1, idata2, idata3):
    """- サンプル構造と不確実性の違いを表示する。
    - 引数: mo, inference_view, idata1, idata2, idata3。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.vstack([mo.md("### 全50件"), idata1, inference_view(idata1, ["mu", "sigma"]),
               mo.md("### 先頭5件\n発散を避けるため、移植版はtarget_accept=0.95で実行します。"), inference_view(idata2, ["mu", "sigma"]),
               mo.md("### 精度tauによるモデル"), inference_view(idata3, ["mu", "sigma"])])
    return


@app.cell
def _(np, plt, sns, stats, df1, X, idata1, posterior_mean):
    """- 正規密度・KDE・観測ヒストグラムを重ねる。
    - 引数: np, plt, sns, stats, df1, X, idata1, posterior_mean。戻り値: mu_mean1, sigma_mean1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mu_mean1 = float(posterior_mean(idata1, "mu"))
    sigma_mean1 = float(posterior_mean(idata1, "sigma"))
    # - ヒストグラムの確率質量と密度の単位を合わせるため、幅0.2を掛ける。
    delta = 0.2
    x_list = np.arange(X.min(), X.max(), 0.01)
    y_list = stats.norm.pdf(x_list, mu_mean1, sigma_mean1)
    _figure, _axis = plt.subplots(figsize=(6, 3))
    sns.histplot(data=df1, x="sepal_length", bins=np.arange(4, 6.2, delta), kde=True,
                 stat="probability", ax=_axis)
    _axis.plot(x_list, y_list * delta, color="blue", label="Posterior mean density")
    _axis.set(title="Observed distribution and fitted normal", xlabel="Sepal length", ylabel="Probability")
    _axis.legend()
    plt.close(_figure)
    _figure
    return (mu_mean1, sigma_mean1,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
