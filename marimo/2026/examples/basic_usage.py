# - 作成日: 2026-09-26
# - 更新日: 2026-09-27。コメントを項目別に整理し、セルの計算と依存を維持する。
# - 目的: 平均・表・図・実行ボタンで、marimoの基本操作を試す。
# - 役割: HowToUse_marimo.mdと実行検証が共有する、学習例の原本。
# - 使用方法: 現在のVS Codeで開き、env.cmd configureで登録済みのPythonを選ぶ。
# - 制約: CPython 3.14.7、marimo 0.25.0。外部データは取得しない。
# - 非対応: 書籍16本の移植、実データの推論、EXE配布。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- ライブラリを準備する。
    - 引数: なし。
    - 戻り値: mo、np、plt、sys。
    - 前提: プロジェクトのNotebook専用環境が準備済み。
    - 副作用: ライブラリのimport。未導入時はImportError。
    - 使用例: このNotebookの最初のセルとして実行する。
    """
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt, sys


@app.cell
def _(sys):
    """- 実行系を確認する。
    - 引数: sys。戻り値: なし。
    - 前提: Notebookのカーネルを選択済み。
    - 副作用: 実行ファイル・Pythonの版を標準出力へ表示する。
    - 例外: Pythonの版・GILが想定と異なる場合はAssertionError。
    - 使用例: Python executableが指定の.venvであることを確認する。
    """
    # - サーバーではなく、セルを実行するPythonの場所・版・GILを確認する。
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    assert sys.version_info[:3] == (3, 14, 7), "Expected Python 3.14.7"
    assert sys._is_gil_enabled(), "Expected GIL-enabled CPython"
    return


@app.cell
def _(mo):
    """- 観測数を選択する。
    - 引数: mo。戻り値: sample_size。
    - 前提: marimoのUIが使用可能。
    - 副作用: スライダーを表示する。不正な設定時はUI作成例外。
    - 使用例: 初期値10から20へ変更して下流の計算を確認する。
    """
    # - 観測数は5～50個、5個刻み。値の読取は別のセルで行う。
    sample_size = mo.ui.slider(start=5, stop=50, step=5, value=10, label="Sample size")
    sample_size
    return (sample_size,)


@app.cell
def _(mo, np, sample_size):
    """- 平均を計算する。
    - 引数: mo、np、sample_size。戻り値: observations、sample_mean。
    - 前提: sample_size.valueが正の整数。
    - 副作用: 平均を画面へ表示する。入力不正時は計算例外。
    - 使用例: 観測数10なら5.5、20なら10.5を表示する。
    """
    # - 単位のない整数1～nをfloat64配列にし、理論値(n+1)/2と比較できる形にする。
    observations = np.arange(1, sample_size.value + 1, dtype=np.float64)
    sample_mean = float(np.mean(observations))
    mo.md(f"Mean: **{sample_mean:.1f}**")
    return observations, sample_mean


@app.cell
def _(mo, observations):
    """- 観測値の表を表示する。
    - 引数: mo、observations。戻り値: sample_table。
    - 前提: observationsが一次元の観測値。
    - 副作用: 表を画面へ表示する。不正なデータはUI作成例外。
    - 使用例: 初期状態ではValue列に1～10が並ぶ。
    """
    # - 計算に使った値をそのまま表にし、入力との対応を確認する。
    sample_table = mo.ui.table({"Value": observations.tolist()})
    sample_table
    return (sample_table,)


@app.cell
def _(np, observations, plt, sample_mean):
    """- 観測値と平均を描く。
    - 引数: np、observations、plt、sample_mean。戻り値: axis、figure。
    - 前提: 数値の観測値と、その平均が計算済み。
    - 副作用: 図を作成し、描画管理の登録を閉じて表示用に返す。
    - 例外: データや描画処理の不備は元の例外を呼出元へ伝える。
    - 使用例: 破線が平均5.5の位置に表示される。
    """
    # - 横軸は1始まりの観測番号、縦軸は観測値。平均の位置を破線で示す。
    figure, axis = plt.subplots(figsize=(6, 3))
    axis.plot(np.arange(1, observations.size + 1), observations, marker="o")
    axis.axhline(sample_mean, color="tab:red", linestyle="--", label="Mean")
    axis.set(xlabel="Observation", ylabel="Value", title="Sample mean")
    axis.legend()
    figure.tight_layout()
    # - 再実行で描画管理用のFigureが蓄積しないよう閉じ、表示用オブジェクトを渡す。
    plt.close(figure)
    figure
    return axis, figure


@app.cell
def _(mo):
    """- 実行ボタンを表示する。
    - 引数: mo。戻り値: run_calculation。
    - 前提: marimoのUIが使用可能。
    - 副作用: ボタンを画面へ表示する。UIの作成失敗時は例外。
    - 使用例: Run calculationを押して次のセルの計算を開始する。
    """
    # - 入力の変更と、計算を始める意思を別々に受け取る。
    run_calculation = mo.ui.run_button(label="Run calculation")
    run_calculation
    return (run_calculation,)


@app.cell
def _(mo, np, observations, run_calculation):
    """- 押下後の平均を表示する。
    - 引数: mo、np、observations、run_calculation。
    - 戻り値: 押下時だけconfirmed_meanを返す。未押下時は返さない。
    - 前提: Auto-runで実行し、観測値が計算済み。
    - 副作用: 結果を画面へ表示する。未押下時はセルと下流を停止する。
    - 例外: 入力不正時は計算例外。
    - 使用例: 初期値で押下すると5.5を表示する。
    """
    # - ボタンが押されていない再実行では、このセルと依存先の計算を止める。
    mo.stop(not run_calculation.value, mo.md("Click Run calculation."))
    # - 実際の重い推論を追加する場合も、この停止判定より後へ置く。
    confirmed_mean = float(np.mean(observations))
    mo.md(f"Confirmed mean: **{confirmed_mean:.1f}**")
    return (confirmed_mean,)


# - Pythonから直接実行された場合は、marimoに実行順と停止条件の制御を任せる。
if __name__ == "__main__":
    app.run()
