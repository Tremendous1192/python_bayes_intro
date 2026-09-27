# - 作成日: 2026-09-27
# - 目的: 5.2 線形回帰のベイズ推論をVS Codeのmarimoで学習する。
# - 役割: notebooks/5_2_線形回帰のベイズ推論.ipynbの対応する学習内容を移植した原本。
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
    """- 観測値と元の3点抽出を準備する。
    - 引数: mo, np, plt。戻り値: X, Y, X_less, Y_less, sample_indexes。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import random
    from notebook_data import load_data
    df = load_data("iris.csv")
    df1 = df.query('species == "versicolor"')
    X = df1["sepal_length"].to_numpy()
    Y = df1["sepal_width"].to_numpy()
    # - 元の抽出方法を局所的な乱数生成器で再現し、他セルの乱数状態を変えない。
    sample_indexes = random.Random(42).sample(range(len(X)), 3)
    X_less, Y_less = X[sample_indexes], Y[sample_indexes]
    _figure, _axes = plt.subplots(1, 2, figsize=(10, 3))
    _axes[0].scatter(X, Y)
    _axes[1].scatter(X_less, Y_less)
    # - 観測の範囲と単位を同じラベルで比較する。
    for _axis in _axes:
        _axis.set(xlabel="Sepal length", ylabel="Sepal width")
    _axes[0].set_title("All 50 observations")
    _axes[1].set_title("Three selected observations")
    plt.close(_figure)
    mo.vstack([mo.md("## 5.2 線形回帰\n全50件と乱数で選んだ3件を比較します。3件ではtarget_accept=0.995と既定0.8の違いも調べます。既定値の例は発散を含む場合があります。"),
               mo.ui.table(df.head()), _figure])
    return (X, Y, X_less, Y_less, sample_indexes,)


@app.cell
def _(pm, X, Y, X_less, Y_less, mo, model_view):
    """- 定義方法と観測数の異なるモデルを準備する。
    - 引数: pm, X, Y, X_less, Y_less, mo, model_view。戻り値: model1, model2, model3, model4。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - Dataを使わない最初の定義もモデル図の比較用に残す。
    with pm.Model() as model1:
        _alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        _beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        _epsilon = pm.HalfNormal("epsilon", sigma=1.0)
        pm.Normal("Y_obs", mu=_alpha * X + _beta, sigma=_epsilon, observed=Y)
    # - ConstantDataに代えて現行のDataを使用し、推論中には値を変更しない。
    with pm.Model() as model2:
        _X_data = pm.Data("X_data", X)
        _Y_data = pm.Data("Y_data", Y)
        _alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        _beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        _mu = pm.Deterministic("mu", _alpha * _X_data + _beta)
        _epsilon = pm.HalfNormal("epsilon", sigma=1.0)
        pm.Normal("obs", mu=_mu, sigma=_epsilon, observed=_Y_data)
    # - 3点でも事前分布と観測モデルは保持する。
    with pm.Model() as model3:
        _X_data = pm.Data("X_data", X_less)
        _Y_data = pm.Data("Y_data", Y_less)
        _alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        _beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        _mu = pm.Deterministic("mu", _alpha * _X_data + _beta)
        _epsilon = pm.HalfNormal("epsilon", sigma=1.0)
        pm.Normal("obs", mu=_mu, sigma=_epsilon, observed=_Y_data)
    # - 未調整例は独立したモデルとし、設定の違いを明示する。
    with pm.Model() as model4:
        _X_data = pm.Data("X_data", X_less)
        _Y_data = pm.Data("Y_data", Y_less)
        _alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)
        _beta = pm.Normal("beta", mu=0.0, sigma=10.0)
        _mu = pm.Deterministic("mu", _alpha * _X_data + _beta)
        _epsilon = pm.HalfNormal("epsilon", sigma=1.0)
        pm.Normal("obs", mu=_mu, sigma=_epsilon, observed=_Y_data)
    mo.vstack([model_view(model1), model_view(model2)])
    return (model1, model2, model3, model4,)


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
def _(mo, run_inference, sampling_mode, sample_model, model2, model3, model4):
    """- 教材の3条件を保持して推論する。
    - 引数: mo, run_inference, sampling_mode, sample_model, model2, model3, model4。戻り値: idata2, idata3, idata4。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Click Run inference."))
    idata2 = sample_model(model2, sampling_mode.value)
    idata3 = sample_model(model3, sampling_mode.value, target_accept=0.995)
    idata4 = sample_model(model4, sampling_mode.value)
    return (idata2, idata3, idata4,)


@app.cell
def _(mo, np, plt, inference_view, idata2, idata3, idata4, X, Y, X_less, Y_less):
    """- 推論の診断と回帰直線の不確実性を比較する。
    - 引数: mo, np, plt, inference_view, idata2, idata3, idata4, X, Y, X_less, Y_less。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    from matplotlib.collections import LineCollection
    comparisons = []
    # - 全サンプルの回帰直線をまとめて描画し、線を間引かず描画負荷を抑える。
    for _label, _idata, _X, _Y in [
        ("All observations", idata2, X, Y),
        ("Three observations: tuned", idata3, X_less, Y_less),
        ("Three observations: default", idata4, X_less, Y_less),
    ]:
        _x_values = np.array([_X.min() - 0.1, _X.max() + 0.1])
        _alphas = _idata["posterior"]["alpha"].values.reshape(-1, 1)
        _betas = _idata["posterior"]["beta"].values.reshape(-1, 1)
        _y_preds = _x_values * _alphas + _betas
        _segments = np.stack([np.broadcast_to(_x_values, _y_preds.shape), _y_preds], axis=-1)
        _figure, _axis = plt.subplots(figsize=(6, 3))
        _axis.add_collection(LineCollection(_segments, color="green", alpha=0.01, linewidth=1))
        _axis.scatter(_X, _Y)
        _axis.set(xlim=(_x_values[0], _x_values[1]), ylim=(1.75, 3.75),
                  title=_label, xlabel="Sepal length", ylabel="Sepal width")
        plt.close(_figure)
        comparisons.append(mo.vstack([mo.md(_label), inference_view(_idata, ["alpha", "beta", "epsilon"]), _figure]))
    mo.vstack(comparisons)
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
