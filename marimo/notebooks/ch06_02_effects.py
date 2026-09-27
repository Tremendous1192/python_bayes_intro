# - 作成日: 2026-09-27
# - 目的: ベイズ重回帰による効果検証をVS Codeのmarimoで学習する。
# - 役割: notebooks/6_2_ベイズ回帰モデルによる効果検証.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 固定環境の推論・描画を準備する。
    - 引数: なし。戻り値: mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, az, load_data。
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
    import arviz as az
    from mod_load_data import load_data
    return (mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, az, load_data,)


@app.cell
def _(mo):
    """- 欠損処理と事前分布の比較を説明する。
    - 引数: mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("""# 6.2 Regression effects
    Compare Normal coefficient priors with a regularized horseshoe prior.
    Remove incomplete rows, then standardize each predictor using sample standard deviation (ddof=1).
    Intervals describe this statistical model; coefficient estimates alone do not establish causation.
    """)
    return


@app.cell
def _(load_data, np, plt, mo):
    """- 元表を破壊せず標準化した101行10列を作る。
    - 引数: load_data, np, plt, mo。戻り値: df, df1, y, raw_X, X, N, D, columns。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    df = load_data("test_scores.csv")
    df1 = df.dropna().astype(float)
    y = df1["score"].copy()
    raw_X = df1.drop(columns="score")
    X = (raw_X - raw_X.mean()) / raw_X.std(ddof=1)
    N, D = X.shape
    columns = X.columns.to_numpy()
    _score_fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.hist(df["score"].dropna(), bins=np.arange(0, 150, 10))
    _ax.set(xlabel="Score", ylabel="Count", title="Observed scores")
    _score_fig.tight_layout()
    plt.close(_score_fig)
    mo.vstack([mo.md(f"Rows: {len(df)}; complete rows: {N}; predictors: {D}"),
               mo.ui.table(df.isna().sum().rename("Missing").reset_index()),
               mo.ui.table(df.describe().reset_index()), _score_fig, mo.ui.table(X.head())])
    return (df, df1, y, raw_X, X, N, D, columns,)


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
def _(pm, X, y, columns, model_view):
    """- 通常の重回帰を入力転置とともに定義する。
    - 引数: pm, X, y, columns, model_view。戻り値: model1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 列名を座標として記録し、係数の表示順序と入力の列順を一致させる。
    with pm.Model(coords={"predictors": columns}) as model1:
        _X_data = pm.Data("X_data", X.T)
        _y_data = pm.Data("y_data", y)
        _alpha = pm.Normal("alpha", mu=0.0, sigma=10.0, dims="predictors")
        _beta = pm.Normal("beta", mu=100.0, sigma=25.0)
        _epsilon = pm.HalfNormal("epsilon", sigma=25.0)
        _mu = pm.Deterministic("mu", _alpha @ _X_data + _beta)
        pm.Normal("obs", mu=_mu, sigma=_epsilon, observed=_y_data)
    model_view(model1)
    return (model1,)


@app.cell
def _(pm, np, X, y, N, D, columns, model_view):
    """- 縮小事前分布の数式と説明変数座標を保持する。
    - 引数: pm, np, X, y, N, D, columns, model_view。戻り値: model2, D0。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    D0 = int(D / 2)
    # - 全体と局所の縮小を持つ元教材のregularized horseshoeを保持する。
    with pm.Model(coords={"predictors": columns}) as model2:
        _X_data = pm.Data("X_data", X.T)
        _y_data = pm.Data("y_data", y)
        _epsilon = pm.HalfNormal("epsilon", sigma=25.0)
        _tau = pm.HalfStudentT("tau", nu=2, sigma=D0 / (D - D0) * _epsilon / np.sqrt(N))
        _lam = pm.HalfStudentT("lam", nu=2, dims="predictors")
        _c2 = pm.InverseGamma("c2", alpha=1, beta=0.1)
        _z = pm.Normal("z", mu=0.0, sigma=1.0, dims="predictors")
        _alpha = pm.Deterministic("alpha", _z * _tau * _lam *
                                  pm.math.sqrt(_c2 / (_c2 + _tau**2 * _lam**2)), dims="predictors")
        _beta = pm.Normal("beta", mu=100.0, sigma=25.0)
        _mu = pm.Deterministic("mu", _alpha @ _X_data + _beta)
        pm.Normal("obs", mu=_mu, sigma=_epsilon, observed=_y_data)
    model_view(model2)
    return (model2, D0,)


@app.cell
def _(model1, model2, sample_model, sampling_mode, run_inference, mo):
    """- 両モデルを同じ反復条件で推論する。
    - 引数: model1, model2, sample_model, sampling_mode, run_inference, mo。戻り値: idata1, idata2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Press Run inference."))
    idata1 = sample_model(model1, sampling_mode.value, target_accept=0.95)
    # - 固定環境の本実行で発散1件を検出したため、縮小モデルだけ受容率を上げる。
    idata2 = sample_model(model2, sampling_mode.value, target_accept=0.99)
    return (idata1, idata2,)


@app.cell
def _(idata1, idata2, inference_view, mo):
    """- 係数と誤差の事後分布・診断を表示する。
    - 引数: idata1, idata2, inference_view, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.vstack([mo.md("## Normal coefficient priors"),
               inference_view(idata1, ["alpha", "beta", "epsilon"]),
               mo.md("## Regularized horseshoe"),
               inference_view(idata2, ["alpha", "beta", "epsilon"])])
    return


@app.cell
def _(idata1, idata2, az, plt, mo):
    """- ArviZ 1の戻り値から50%・94% HDIの図を表示する。
    - 引数: idata1, idata2, az, plt, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _forests = []
    # - 通常回帰の結合・チェーン別と縮小回帰の結合を元教材どおり比較する。
    for _title, _idata, _combined in [
        ("Normal priors: combined", idata1, True),
        ("Normal priors: by chain", idata1, False),
        ("Regularized horseshoe: combined", idata2, True),
    ]:
        _collection = az.plot_forest(_idata, var_names=["alpha"], combined=_combined,
                                     ci_kind="hdi", ci_probs=[0.5, 0.94], backend="matplotlib")
        _figure = _collection.get_viz("figure")
        _figure.suptitle(_title)
        _figure.tight_layout()
        plt.close(_figure)
        _forests.append(_figure)
    mo.vstack(_forests)
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
