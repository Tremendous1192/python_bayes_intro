# - 作成日: 2026-09-27
# - 目的: 3クラスの潜在変数をVS Codeのmarimoで学習する。
# - 役割: sample-notebooks/A_3クラス潜在変数モデル.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 固定環境の推論・描画を準備する。
    - 引数: なし。戻り値: mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, sns, pd, load_data, three_class_model。
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
    import seaborn as sns
    import pandas as pd
    from mod_load_data import load_data
    from model_three_class import three_class_model
    return (mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, sns, pd, load_data, three_class_model,)


@app.cell
def _(mo):
    """- カテゴリカル・ディリクレ分布とモデルの比較目的を示す。
    - 引数: mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("""# Three-class latent variable models
    Categorical observations use labels 0, 1, 2. Dirichlet probabilities sum to one.
    Compare precision priors, standard-deviation priors, and ordered component means.
    Component numbers are not species labels. One chain cannot estimate between-chain R-hat.
    """)
    return


@app.cell
def _(pm, np, plt, mo, pd):
    """- 事前分布を500回ずつサンプリングする。
    - 引数: pm, np, plt, mo, pd。戻り値: model1, model2, prior_samples1, prior_samples2, x_samples1, x_samples2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - カテゴリカル分布の固定確率を単独で確認する。
    with pm.Model() as model1:
        pm.Categorical("x", p=[0.2, 0.5, 0.3])
        prior_samples1 = pm.sample_prior_predictive(draws=500, random_seed=42)
    # - 一様なディリクレ分布は単体上にある3要素の確率を生成する。
    with pm.Model() as model2:
        pm.Dirichlet("p", a=np.ones(3))
        prior_samples2 = pm.sample_prior_predictive(draws=500, random_seed=42)
    x_samples1 = prior_samples1["prior"]["x"].values.reshape(-1)
    x_samples2 = prior_samples2["prior"]["p"].values.reshape(-1, 3)
    _prior_fig, _axes = plt.subplots(1, 2, figsize=(10, 4))
    _axes[0].hist(x_samples1, bins=np.arange(-0.5, 3.0, 1.0), rwidth=0.7)
    _axes[0].set(xticks=[0, 1, 2], xlabel="Category", ylabel="Count",
                 title="Categorical: p=(0.2, 0.5, 0.3)")
    _axes[1].scatter(x_samples2[:, 0], x_samples2[:, 1], s=5)
    _axes[1].set(xlabel="p[0]", ylabel="p[1]", title="Dirichlet: a=(1, 1, 1)")
    _prior_fig.tight_layout()
    plt.close(_prior_fig)
    mo.vstack([_prior_fig, mo.ui.table(pd.DataFrame(x_samples2[:10], columns=["p[0]", "p[1]", "p[2]"]))])
    return (model1, model2, prior_samples1, prior_samples2, x_samples1, x_samples2,)


@app.cell
def _(load_data, mo):
    """- 3種150観測を順序を保って読み込む。
    - 引数: load_data, mo。戻り値: df, X。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    df = load_data("iris.csv")
    X = df["petal_width"].to_numpy()
    mo.vstack([mo.ui.table(df.head()), mo.ui.table(df["species"].value_counts().reset_index())])
    return (df, X,)


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
def _(X, three_class_model, model_view, mo):
    """- 潜在カテゴリとパラメータの対応を可視化する。
    - 引数: X, three_class_model, model_view, mo。戻り値: model3, model4, model5。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    model3 = three_class_model(X)
    model4 = three_class_model(X, scale_prior=True)
    model5 = three_class_model(X, ordered=True)
    mo.vstack([model_view(model3), model_view(model4), model_view(model5)])
    return (model3, model4, model5,)


@app.cell
def _(model3, model4, model5, sample_model, sampling_mode, run_inference, mo):
    """- 書籍の1・1・4チェーンを逐次実行する。
    - 引数: model3, model4, model5, sample_model, sampling_mode, run_inference, mo。戻り値: idata3, idata4, idata5。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Press Run inference."))
    idata3 = sample_model(model3, sampling_mode.value, chains=1, draws=2000, target_accept=0.99)
    idata4 = sample_model(model4, sampling_mode.value, chains=1, draws=2000, target_accept=0.99)
    idata5 = sample_model(model5, sampling_mode.value, target_accept=0.99)
    return (idata3, idata4, idata5,)


@app.cell
def _(idata3, idata4, idata5, inference_view, mo):
    """- 失敗例も含めてトレースと診断表を表示する。
    - 引数: idata3, idata4, idata5, inference_view, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.vstack([
        mo.md("## Precision prior"),
        inference_view(idata3, ["p", "mus", "sigmas"]),
        mo.md("## Standard-deviation prior: comparison example"),
        inference_view(idata4, ["p", "mus", "sigmas"]),
        mo.md("## Ordered component means"),
        inference_view(idata5, ["p", "mus", "sigmas"]),
    ])
    return


@app.cell
def _(idata3, idata5, posterior_mean, df, np, plt, stats, sns, mo):
    """- 元教材の度数ヒストグラムと正規曲線を比較する。
    - 引数: idata3, idata5, posterior_mean, df, np, plt, stats, sns, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _figures = []
    # - 制約の前後で成分の番号を種名に対応させず比較する。
    for _title, _idata in [("Precision prior", idata3), ("Ordered means", idata5)]:
        _mus = posterior_mean(_idata, "mus")
        _sigmas = posterior_mean(_idata, "sigmas")
        _x = np.linspace(0.0, 3.0, 200)
        _fig, _ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x="petal_width", hue="species", kde=True,
                     bins=np.arange(0.0, 3.0, 0.1), ax=_ax)
        # - 推論成分の凡例を追加した後も、観測の種別を読み取れるようにする。
        _species_legend = _ax.get_legend()
        _species_legend.set_title("Observed species")
        _species_legend.set_loc("upper right")
        _ax.add_artist(_species_legend)
        # - 係数5は150観測×幅0.1÷3の等比率参照であり、推論したpではない。
        for _component in range(3):
            _ax.plot(_x, stats.norm.pdf(_x, _mus[_component], _sigmas[_component]) * 5,
                     linestyle="--", label=f"Component {_component} (equal-weight reference)")
        _ax.set(title=_title, xlabel="Petal width", ylabel="Count per bin")
        _ax.legend(fontsize=8, loc="upper left")
        _fig.tight_layout()
        plt.close(_fig)
        _figures.append(_fig)
    mo.vstack(_figures)
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
