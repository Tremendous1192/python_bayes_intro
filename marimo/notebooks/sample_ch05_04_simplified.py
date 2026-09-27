# - 作成日: 2026-09-27
# - 目的: 混合確率を固定した潜在変数をVS Codeのmarimoで学習する。
# - 役割: sample-notebooks/5_4_潜在変数モデル_簡略版.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 固定環境の推論・描画を準備する。
    - 引数: なし。戻り値: mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, sns, load_data, two_class_model。
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
    from mod_load_data import load_data
    from model_two_class import two_class_model
    return (mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, sns, load_data, two_class_model,)


@app.cell
def _(mo):
    """- 教材のモデルとラベルの解釈を説明する。
    - 引数: mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("""# Latent variables: fixed mixture probability
    Run the precision-prior model and the standard-deviation-prior comparison.
    The mixture probability is fixed at 0.5; the first run uses seed 1.
    Component numbers are latent labels, not verified species labels.
    A single chain cannot estimate between-chain R-hat.
    """)
    return


@app.cell
def _(load_data, np, plt, sns, mo):
    """- 固定した2種100件の観測を確認する。
    - 引数: load_data, np, plt, sns, mo。戻り値: df, df2, X, bins。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    df = load_data("iris.csv")
    df2 = df.loc[df["species"] != "setosa"].reset_index(drop=True)
    X = df2["petal_width"].to_numpy()
    bins = np.arange(0.8, 3.0, 0.1)
    _observed, _axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(x=X, bins=bins, ax=_axes[0])
    sns.histplot(data=df2, x="petal_width", hue="species", bins=bins, kde=True, ax=_axes[1])
    # - すべての観測を同一の横軸で比較する。
    for _ax in _axes:
        _ax.set(xlabel="Petal width", ylabel="Count")
    _observed.tight_layout()
    plt.close(_observed)
    mo.vstack([mo.ui.table(df2.head()), _observed])
    return (df, df2, X, bins,)


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
def _(X, two_class_model, model_view, mo):
    """- 精度・標準偏差の事前分布を明示して比較する。
    - 引数: X, two_class_model, model_view, mo。戻り値: model1, model2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    model1 = two_class_model(X, fixed_probability=True)
    model2 = two_class_model(X, fixed_probability=True, scale_prior=True)
    mo.vstack([model_view(model1), model_view(model2)])
    return (model1, model2,)


@app.cell
def _(model1, model2, sample_model, sampling_mode, run_inference, mo):
    """- クリック時だけ書籍の反復条件で推論する。
    - 引数: model1, model2, sample_model, sampling_mode, run_inference, mo。戻り値: idata1, idata2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Press Run inference."))
    idata1 = sample_model(model1, sampling_mode.value, chains=1, draws=2000, target_accept=0.99, seed=1)
    idata2 = sample_model(model2, sampling_mode.value, chains=1, target_accept=0.998)
    return (idata1, idata2,)


@app.cell
def _(idata1, idata2, inference_view, mo):
    """- 事後分布・トレース・診断表を省略せず示す。
    - 引数: idata1, idata2, inference_view, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.vstack([
        mo.md("## Precision prior"),
        inference_view(idata1, ["mus", "sigmas"]),
        mo.md("## Standard-deviation prior: comparison example"),
        inference_view(idata2, ["mus", "sigmas"])
    ])
    return


@app.cell
def _(idata1, posterior_mean, df2, bins, np, stats, plt, sns, mo):
    """- 平均パラメータの正規分布を、ラベルの断定なしで重ねる。
    - 引数: idata1, posterior_mean, df2, bins, np, stats, plt, sns, mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _overlays = []
    # - 成分番号を種名に固定せず、観測の等しい種別比率を参照として表示する。
    for _title, _idata in [("Precision prior", idata1)]:
        _mus = posterior_mean(_idata, "mus")
        _sigmas = posterior_mean(_idata, "sigmas")
        _x = np.linspace(0.8, 3.0, 200)
        _fig, _ax = plt.subplots(figsize=(7, 4))
        sns.histplot(data=df2, x="petal_width", hue="species", bins=bins,
                     stat="probability", kde=True, ax=_ax)
        # - 後から成分の凡例を追加しても、観測の種別凡例が消えないよう保持する。
        _species_legend = _ax.get_legend()
        _species_legend.set_title("Observed species")
        _species_legend.set_loc("upper right")
        _ax.add_artist(_species_legend)
        # - 曲線の0.1/2はヒストグラム幅と等しい参照比率であり、推定pではない。
        for _component in range(2):
            _ax.plot(_x, stats.norm.pdf(_x, _mus[_component], _sigmas[_component]) * 0.1 / 2,
                     linestyle="--", label=f"Component {_component} (equal-weight reference)")
        _ax.set(title=_title, xlabel="Petal width", ylabel="Probability per bin")
        _ax.legend(fontsize=8, loc="upper left")
        _fig.tight_layout()
        plt.close(_fig)
        _overlays.append(_fig)
    mo.vstack(_overlays)
    return


@app.cell
def _(df2, idata1, np, plt, mo):
    """- 選んだ5観測の潜在ラベルの不確実性を表示する。
    - 引数: df2, idata1, np, plt, mo。戻り値: value_list, indexes, df_heads, sval。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    value_list = [1.0, 1.5, 1.7, 2.0, 2.5]
    # - 各花弁幅に一致する最初の観測番号を保ち、sの観測軸と対応させる。
    indexes = [int(np.flatnonzero(df2["petal_width"].to_numpy() == _value)[0])
               for _value in value_list]
    df_heads = df2.loc[indexes]
    _s = idata1["posterior"]["s"]
    sval = _s.values[:, :, indexes].reshape(-1, len(indexes)).T
    _latent, _axes = plt.subplots(1, 5, figsize=(13, 3))
    # - 観測ごとの潜在ラベルの頻度を同じ縦軸上限で比較する。
    for _ax, _item, _value, _index in zip(_axes, sval, value_list, indexes):
        _ax.hist(_item, bins=[-0.5, 0.5, 1.5], rwidth=0.7)
        _ax.set(xticks=[0, 1], ylim=(0, _item.size),
                title=f"Width {_value}\nIndex {_index}", xlabel="Latent label")
    _latent.tight_layout()
    plt.close(_latent)
    mo.vstack([mo.ui.table(df_heads), _latent])
    return (value_list, indexes, df_heads, sval,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
