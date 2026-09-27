# - 作成日: 2026-09-27
# - 目的: 第3章 ベイズ推論の考え方をVS Codeのmarimoで学習する。
# - 役割: notebooks/3章_ベイズ推論とは.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 元Notebookの一様事前分布の例を説明する。
    - 引数: なし。戻り値: np, plt。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    mo.md("## 第3章 ベイズ推論とは\n観測前の知識を事前分布で表します。ここでは区間[0, 1]の一様密度を確認し、第4章で観測値による更新を実装します。")
    return (np, plt,)


@app.cell
def _(np, plt):
    """- 事前分布の面積が1になることを描画する。
    - 引数: np, plt。戻り値: coordinates, density。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    def f(x):
        """区間内の一様密度を返す。引数は0から1の座標、戻り値は同形状の1。
        前提: 座標は区間内。副作用・独自例外なし。例: f(np.array([0.5]))。
        """
        return x - x + 1.0

    coordinates = np.linspace(0.0, 1.0, 101)
    density = f(coordinates)
    figure, axis = plt.subplots(figsize=(6, 3))
    axis.fill_between(coordinates, density)
    axis.set(xlabel="Probability", ylabel="Density", title="Uniform prior on [0, 1]", ylim=(0, 1.2))
    plt.close(figure)
    figure
    return (coordinates, density,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
