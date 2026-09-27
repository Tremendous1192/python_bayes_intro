# - 作成日: 2026-09-27
# - 目的: 潜在変数のチェーン別診断をVS Codeのmarimoで学習する。
# - 役割: sample-notebooks/FAQ_潜在変数モデル.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 固定環境の推論・描画を準備する。
    - 引数: なし。戻り値: mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, load_data, two_class_model。
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
    from mod_load_data import load_data
    from model_two_class import two_class_model
    return (mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, load_data, two_class_model,)


@app.cell
def _(mo):
    """- チェーン番号とラベルの対応が固定ではないことを明記する。
    - 引数: mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("""# FAQ: label switching across five chains
    Compare chains [0, 1], [2], and [3, 4], as in the original example.
    The chain that switches labels can change with the PyMC version and random stream.
    Inspect all five chains before interpreting the posterior means.
    """)
    return


@app.cell
def _(load_data):
    """- FAQと本編で同じ観測順序を使う。
    - 引数: load_data。戻り値: df, df2, X。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    df = load_data("iris.csv")
    df2 = df.loc[df["species"] != "setosa"].reset_index(drop=True)
    X = df2["petal_width"].to_numpy()
    return (df, df2, X,)


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
def _(X, two_class_model, model_view):
    """- 本編と同じ2クラス精度モデルを作る。
    - 引数: X, two_class_model, model_view。戻り値: model1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    model1 = two_class_model(X)
    model_view(model1)
    return (model1,)


@app.cell
def _(model1, sample_model, sampling_mode, run_inference, mo):
    """- FAQの5チェーンを1コアで順番に推論する。
    - 引数: model1, sample_model, sampling_mode, run_inference, mo。戻り値: idata1_2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Press Run inference."))
    idata1_2 = sample_model(model1, sampling_mode.value, chains=5, target_accept=0.99)
    return (idata1_2,)


@app.cell
def _(idata1_2, inference_view, mo):
    """- 全体診断と3つのチェーングループを比較する。
    - 引数: idata1_2, inference_view, mo。戻り値: chain_groups。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    chain_groups = [[0, 1], [2], [3, 4]]
    _views = [mo.md("## All five chains"), inference_view(idata1_2, ["p", "mus", "sigmas"])]
    # - 元教材のグループを保持し、単鎖ではR-hatを評価できないことを示す。
    for _chains in chain_groups:
        _views.extend([mo.md(f"## Chains {_chains}"),
                       inference_view(idata1_2, ["p", "mus", "sigmas"], coords={"chain": _chains})])
    mo.vstack(_views)
    return (chain_groups,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
