# - 作成日: 2026-09-27
# - 目的: 5.3 階層ベイズモデルをVS Codeのmarimoで学習する。
# - 役割: notebooks/5_3_階層ベイズモデル.ipynbの対応する学習内容を移植した原本。
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
def _(mo, np, plt):
    """- 9観測とカテゴリ対応を準備する。
    - 引数: mo, np, plt。戻り値: df, df_sel, X, Y, cl, species_order, sns。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import pandas as pd
    import random
    import seaborn as sns
    from mod_load_data import load_data
    df = load_data("iris.csv")
    species_order = ["setosa", "versicolor", "virginica"]
    sample_indexes = random.Random(42).sample(range(50), 3)
    # - 各種から同じ3インデックスを選び、元の計9件を再現する。
    df_sel = pd.concat([df.loc[df["species"] == species].iloc[sample_indexes] for species in species_order], ignore_index=True)
    X = df_sel["sepal_length"].to_numpy()
    Y = df_sel["sepal_width"].to_numpy()
    cl = pd.Categorical(df_sel["species"], categories=species_order).codes.astype("int64")
    _figure, _axis = plt.subplots(figsize=(6, 3))
    sns.scatterplot(data=df_sel, x="sepal_length", y="sepal_width", hue="species", style="species", ax=_axis)
    _axis.set_title("Nine selected observations")
    plt.close(_figure)
    mo.vstack([mo.md('## 5.3 Hierarchical Bayesian model\nSpecies-specific slopes and intercepts share hyperdistributions. Keep the category order fixed to match estimates with species names.'),
               mo.ui.table(df_sel), _figure])
    return (df, df_sel, X, Y, cl, species_order, sns,)


@app.cell
def _(pm, X, Y, cl, mo, model_view):
    """- 階層事前分布と観測モデルを定義する。
    - 引数: pm, X, Y, cl, mo, model_view。戻り値: model1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 上位分布の平均・標準偏差から、3種類の回帰係数を生成する。
    with pm.Model() as model1:
        _X_data = pm.Data("X_data", X)
        _Y_data = pm.Data("Y_data", Y)
        _cl_data = pm.Data("cl_data", cl)
        _a_mu = pm.Normal("a_mu", mu=0.0, sigma=10.0)
        _a_sigma = pm.HalfNormal("a_sigma", sigma=10.0)
        _alpha = pm.Normal("alpha", mu=_a_mu, sigma=_a_sigma, shape=3)
        _b_mu = pm.Normal("b_mu", mu=0.0, sigma=10.0)
        _b_sigma = pm.HalfNormal("b_sigma", sigma=10.0)
        _beta = pm.Normal("beta", mu=_b_mu, sigma=_b_sigma, shape=3)
        _epsilon = pm.HalfNormal("epsilon", sigma=1.0)
        _mu = pm.Deterministic("mu", _X_data * _alpha[_cl_data] + _beta[_cl_data])
        pm.Normal("obs", mu=_mu, sigma=_epsilon, observed=_Y_data)
    mo.vstack([model_view(model1), mo.md('Data supplies observations; Deterministic stores mu.')])
    return (model1,)


@app.cell
def _(mo, np):
    """- 書籍の配列インデックスの例を保持する。
    - 引数: mo, np。戻り値: MU。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 添字による種ごとのパラメータ選択を推論と独立して確認する。
    ALPHA = np.array([0.1, 0.2, 0.3])
    CL = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    MU = ALPHA[CL]
    mo.ui.table({"Class": CL, "Selected coefficient": MU})
    return (MU,)


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
def _(mo, run_inference, sampling_mode, sample_model, model1):
    """- 元の高い受容率を保持して階層モデルを推論する。
    - 引数: mo, run_inference, sampling_mode, sample_model, model1。戻り値: idata1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Click Run inference."))
    idata1 = sample_model(model1, sampling_mode.value, target_accept=0.998)
    return (idata1,)


@app.cell
def _(mo, np, plt, sns, df, df_sel, species_order, posterior_mean, idata1, inference_view):
    """- 部分プーリングの結果を観測値と比較する。
    - 引数: mo, np, plt, sns, df, df_sel, species_order, posterior_mean, idata1, inference_view。戻り値: alpha_means, beta_means。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    alpha_means = posterior_mean(idata1, "alpha").values
    beta_means = posterior_mean(idata1, "beta").values
    _figures = []
    # - 選択した9件と全150件の両方に同じ推論結果を重ねる。
    for _label, _frame in [("Selected observations", df_sel), ("All observations", df)]:
        _x = np.array([_frame["sepal_length"].min() - 0.1, _frame["sepal_length"].max() + 0.1])
        _figure, _axis = plt.subplots(figsize=(6, 3))
        sns.scatterplot(data=_frame, x="sepal_length", y="sepal_width", hue="species", ax=_axis)
        # - 種の順序を明示した係数で回帰直線を描く。
        for _index, _species in enumerate(species_order):
            _axis.plot(_x, alpha_means[_index] * _x + beta_means[_index], label=_species)
        _axis.set_title(_label)
        _axis.legend()
        plt.close(_figure)
        _figures.append(_figure)
    mo.vstack([inference_view(idata1, ["alpha", "beta"]), *_figures])
    return (alpha_means, beta_means,)


@app.cell
def _(pm, X, Y, cl, mo, model_view):
    """- 数式を保持してモデル図の構成要素を比較する。
    - 引数: pm, X, Y, cl, mo, model_view。戻り値: model2, model3。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - コラムのDataなしモデルでは、muを保存しない定義を比較する。
    with pm.Model() as model2:
        _a_mu = pm.Normal("a_mu", mu=0.0, sigma=10.0)
        _a_sigma = pm.HalfNormal("a_sigma", sigma=10.0)
        _alpha = pm.Normal("alpha", mu=_a_mu, sigma=_a_sigma, shape=3)
        _b_mu = pm.Normal("b_mu", mu=0.0, sigma=10.0)
        _b_sigma = pm.HalfNormal("b_sigma", sigma=10.0)
        _beta = pm.Normal("beta", mu=_b_mu, sigma=_b_sigma, shape=3)
        _epsilon = pm.HalfNormal("epsilon", sigma=1.0)
        pm.Normal("Y_obs", mu=X * _alpha[cl] + _beta[cl], sigma=_epsilon, observed=Y)
    # - 元のmodel3も独立して定義し、Data・Deterministicの役割を比較する。
    with pm.Model() as model3:
        _X_data = pm.Data("X_data", X)
        _Y_data = pm.Data("Y_data", Y)
        _cl_data = pm.Data("cl_data", cl)
        _a_mu = pm.Normal("a_mu", mu=0.0, sigma=10.0)
        _a_sigma = pm.HalfNormal("a_sigma", sigma=10.0)
        _alpha = pm.Normal("alpha", mu=_a_mu, sigma=_a_sigma, shape=3)
        _b_mu = pm.Normal("b_mu", mu=0.0, sigma=10.0)
        _b_sigma = pm.HalfNormal("b_sigma", sigma=10.0)
        _beta = pm.Normal("beta", mu=_b_mu, sigma=_b_sigma, shape=3)
        _epsilon = pm.HalfNormal("epsilon", sigma=1.0)
        _mu = pm.Deterministic("mu", _X_data * _alpha[_cl_data] + _beta[_cl_data])
        pm.Normal("obs", mu=_mu, sigma=_epsilon, observed=_Y_data)
    mo.vstack([mo.md('### Model components'), model_view(model2), model_view(model3)])
    return (model2, model3,)


@app.cell
def _(mo):
    """- 元のExcel出力を明示操作にする。
    - 引数: mo。戻り値: export_table。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    export_table = mo.ui.run_button(label="Save Excel preview")
    export_table
    return (export_table,)


@app.cell
def _(df, export_table, mo):
    """- 先頭5件をプロジェクト内へExcel保存する。
    - 引数: df, export_table, mo。戻り値: excel_path。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import time
    from mod_load_data import DATA_ROOT
    mo.stop(not export_table.value, mo.md("Click Save Excel preview."))
    _folder = DATA_ROOT.parent / ".cache" / "exports"
    _folder.mkdir(parents=True, exist_ok=True)
    excel_path = _folder / f"iris_preview_{time.time_ns()}.xlsx"
    # - 固定範囲へ新規作成し、既存の利用者ファイルを上書きしない。
    with excel_path.open("xb") as _handle:
        df.head().to_excel(_handle, engine="openpyxl")
    mo.md(f"Saved: `{excel_path}`")
    return (excel_path,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
