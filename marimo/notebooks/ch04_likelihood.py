# - 作成日: 2026-09-27
# - 目的: 第4章 4.2 最尤推定をVS Codeのmarimoで学習する。
# - 役割: notebooks/4章_はじめてのベイズ推論実習.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 最尤推定の問題を定義する。
    - 引数: なし。戻り値: mo, np, plt, torch。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    mo.md('## 4.2 Maximum likelihood\nThe likelihood for 2 successes in 5 trials is p²(1−p)³. Compare the analytic estimate 2/5 with automatic differentiation.')
    return (mo, np, plt, torch,)


@app.cell
def _(np, torch):
    """- 元の自動微分による最尤推定を保持する。
    - 引数: np, torch。戻り値: optimization_log, maximum_likelihood。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    def log_lh(p):
        """対数尤度を返す。引数は0<p<1のTensor、戻り値はスカラーTensor。
        前提: torch導入済み。副作用なし。範囲外は非有限値。例: log_lh(torch.tensor(0.4))。
        """
        return 2 * torch.log(p) + 3 * torch.log1p(-p)

    p = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)
    logs = []
    # - 書籍と同じ40回・学習率0.01で損失を最小化する。
    for epoch in range(40):
        loss = -log_lh(p)
        loss.backward()
        # - 更新自体を微分グラフへ入れず、次の反復の勾配を初期化する。
        with torch.no_grad():
            p -= 0.01 * p.grad
            p.grad.zero_()
        logs.append((epoch, p.item(), loss.item()))
    optimization_log = np.asarray(logs)
    maximum_likelihood = p.item()
    return (optimization_log, maximum_likelihood,)


@app.cell
def _(mo, np, plt, optimization_log, maximum_likelihood):
    """- 尤度・更新値・損失を比較する。
    - 引数: mo, np, plt, optimization_log, maximum_likelihood。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    _x = np.linspace(0.001, 0.999, 300)
    _figure, _axes = plt.subplots(1, 3, figsize=(12, 3))
    _axes[0].plot(_x, _x ** 2 * (1 - _x) ** 3)
    _axes[0].set(title="Likelihood", xlabel="p")
    _axes[1].plot(optimization_log[:, 0], optimization_log[:, 1])
    _axes[1].axhline(0.4, linestyle="--", color="black")
    _axes[1].set(title="Estimated p", xlabel="Epoch")
    _axes[2].plot(optimization_log[:, 0], optimization_log[:, 2])
    _axes[2].set(title="Loss", xlabel="Epoch")
    plt.close(_figure)
    mo.vstack([mo.md(f"Estimated p: **{maximum_likelihood:.6f}**"), _figure])
    return


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    app.run()
