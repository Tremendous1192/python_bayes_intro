# - 作成日: 2026-09-27
# - 目的: 参考 第4章 図4.4用をVS Codeのmarimoで学習する。
# - 役割: sample-notebooks/4章_図4_4用.ipynbの対応する学習内容を移植した原本。
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
def _(mo, pm, model_view):
    """- 元の2モデルを区別して定義する。
    - 引数: mo, pm, model_view。戻り値: prediction_model, nested_model。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _introduction = mo.md('## Supplement for Figure 4.4\nThe unobserved discrete variable Y_pred requires a PyMC compound step. Save the PDF locally using the export button.')
    # - 予測成功数を未知変数として追加する元のモデルを保持する。
    with pm.Model() as prediction_model:
        _p = pm.Uniform("p", lower=0.0, upper=1.0)
        pm.Binomial("X_obs", p=_p, n=50, observed=20)
        pm.Binomial("Y_pred", p=_p, n=1000)
    # - 予測成功数から別の成功確率を構成する補足モデルを保持する。
    with pm.Model() as nested_model:
        _p = pm.Uniform("p", lower=0.0, upper=1.0)
        _y = pm.Binomial("Y_pred", p=_p, n=1000)
        _p2 = pm.Beta("p2", alpha=_y + 1, beta=1001 - _y)
        pm.Binomial("X_obs", p=_p2, n=50, observed=20)
    mo.vstack([_introduction, model_view(prediction_model), model_view(nested_model)])
    return (prediction_model, nested_model,)


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
def _(mo, run_inference, sampling_mode, sample_model, prediction_model, nested_model):
    """- 離散・連続変数の複合サンプリングを実行する。
    - 引数: mo, run_inference, sampling_mode, sample_model, prediction_model, nested_model。戻り値: prediction_result, nested_result。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Click Run inference."))
    prediction_result = sample_model(prediction_model, sampling_mode.value)
    nested_result = sample_model(nested_model, sampling_mode.value, target_accept=0.999)
    return (prediction_result, nested_result,)


@app.cell
def _(mo, inference_view, prediction_result, nested_result):
    """- pだけと全未知変数の表示を比較する。
    - 引数: mo, inference_view, prediction_result, nested_result。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.vstack([inference_view(prediction_result, ["p"]),
               inference_view(prediction_result, ["p", "Y_pred"]),
               inference_view(nested_result, ["p", "p2", "Y_pred"])])
    return


@app.cell
def _(mo):
    """- ローカルPDF保存の操作を用意する。
    - 引数: mo。戻り値: export_pdf。前提: marimo起動済み。
    - 副作用: ボタン表示。UI失敗は元の例外。例: Save PDFを押す。
    """
    export_pdf = mo.ui.run_button(label="Save PDF")
    export_pdf
    return (export_pdf,)


@app.cell
def _(export_pdf, mo, nested_result, plt):
    """- 図をプロジェクト内へ保存する。
    - 引数: export_pdf、mo、nested_result、plt。戻り値: pdf_path。
    - 前提: 推論済み。副作用: 新規PDFの保存。失敗はI/O例外。
    - 使用例: Save PDFの後、表示された保存先を開く。
    """
    import time
    import arviz as az
    from mod_load_data import DATA_ROOT
    mo.stop(not export_pdf.value, mo.md("Click Save PDF to export the trace."))
    _collection = az.plot_trace_dist(nested_result, var_names=["p", "p2", "Y_pred"],
                                     compact=False, backend="matplotlib")
    _figure = _collection.get_viz("figure")
    _folder = DATA_ROOT.parent / ".cache" / "exports"
    _folder.mkdir(parents=True, exist_ok=True)
    pdf_path = _folder / f"figure_4_4_{time.time_ns()}.pdf"
    # - 元のColabダウンロードを固定範囲への排他作成に置き換える。
    with pdf_path.open("xb") as _handle:
        _figure.savefig(_handle, format="pdf")
    plt.close(_figure)
    mo.md(f"Saved: `{pdf_path}`")
    return (pdf_path,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
