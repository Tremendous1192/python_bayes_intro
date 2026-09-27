# - 作成日: 2026-09-27
# - 目的: IRTによるテスト結果評価とADVIをVS Codeのmarimoで学習する。
# - 役割: notebooks/6_3_IRTによるテスト結果評価.ipynbの対応する学習内容を移植した原本。
# - 使用: 登録済み.venvを選び、このファイルをmarimo拡張機能で開く。
# - 制約: 固定依存、CPU、Notebook専用。データを使う教材は同梱CSVを読み込む。
# - 非対応: Colab、セルからの依存導入、EXE・GPU実行。
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    """- 固定環境の推論・描画を準備する。
    - 引数: なし。戻り値: mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, pd, az, expit, load_data, irt_model, ability_summary。
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
    import pandas as pd
    import arviz as az
    from scipy.special import expit
    from mod_load_data import load_data
    from model_irt import irt_model, ability_summary
    return (mo, np, plt, pm, stats, sample_model, posterior_mean, inference_view, model_view, pd, az, expit, load_data, irt_model, ability_summary,)


@app.cell
def _(mo):
    """- 全データの推論条件と長時間実行を説明する。
    - 引数: mo。戻り値: なし。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.md("""# 6.3 Item response theory
    Use all 1,000 users and 50 questions to fit the two-parameter logistic model.
    Book run: MCMC 4 × 1,000 retained draws, then ADVI 20,000 iterations and 2,000 draws.
    This is a long CPU run. Quick check only checks wiring; it is not a convergence assessment.
    Stored posterior variables are theta, a, b. Per-response logits are not needed for these analyses.
    """)
    return


@app.cell
def _(expit, np, pd, plt, mo):
    """- 2PLの項目特性曲線を先に確認する。
    - 引数: expit, np, pd, plt, mo。戻り値: f, params, x, vals。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    def f(x):
        """- シグモイド。引数: 実数または配列。戻り値: 0から1の確率。
        - 前提: 有限値。副作用: なし。失敗: SciPy例外。使用例: f(0.0)。
        """
        return expit(x)

    params = [(1, 0), (2, 0), (1, 2), (2, 2)]
    x = np.arange(-4, 4.1, 0.1)
    vals = []
    _curve, _ax = plt.subplots(figsize=(7, 4))
    # - 識別力と困難度を別々に変え、能力値に対する正答確率を比較する。
    for _a, _b in params:
        _ax.plot(x, f(_a * (x - _b)), label=f"a={_a}, b={_b}")
        vals.append([_a, _b, f(_a * (1 - _b)), f(_a * (2 - _b))])
    _ax.set(xlabel="Ability", ylabel="Probability of a correct answer",
            title="Item characteristic curves")
    _ax.legend()
    _curve.tight_layout()
    plt.close(_curve)
    mo.vstack([_curve, mo.ui.table(pd.DataFrame(vals, columns=["a", "b", "f(1)", "f(2)"]))])
    return (f, params, x, vals,)


@app.cell
def _(load_data, pd, mo):
    """- 固定CSVを縦持ちにし、名前と整数番号の対応を保持する。
    - 引数: load_data, pd, mo。戻り値: df, response_df, user_idx, users, question_idx, questions, response。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    df = load_data("irt-sample.csv")
    response_df = df.rename_axis("user").reset_index().melt(
        id_vars="user", var_name="question", value_name="response")
    user_idx, users = pd.factorize(response_df["user"], sort=False)
    question_idx, questions = pd.factorize(response_df["question"], sort=False)
    response = response_df["response"].to_numpy()
    mo.vstack([mo.md(f"Response matrix: {df.shape}; long table: {response_df.shape}"),
               mo.ui.table(df.head()), mo.ui.table(response_df.head())])
    return (df, response_df, user_idx, users, question_idx, questions, response,)


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
def _(response, user_idx, question_idx, users, questions, irt_model, model_view):
    """- 受験者と設問の名前付きモデルを作る。
    - 引数: response, user_idx, question_idx, users, questions, irt_model, model_view。戻り値: model1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    model1 = irt_model(response, user_idx, question_idx, users, questions)
    model_view(model1)
    return (model1,)


@app.cell
def _(model1, sample_model, sampling_mode, run_inference, mo):
    """- 元のMCMC条件を保持し、保存変数だけを限定する。
    - 引数: model1, sample_model, sampling_mode, run_inference, mo。戻り値: idata1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value, mo.md("Press Run inference."))
    # - 5万個のlogitを4000回保存する約1.6GBの不要な配列を避ける。
    # - 全観測でCと密度・勾配の一致を検証済みのNumbaを使い、長い評価時間を抑える。
    idata1 = sample_model(model1, sampling_mode.value, var_names=["theta", "a", "b"], backend="numba")
    return (idata1,)


@app.cell
def _(idata1, inference_view, az, mo):
    """- 代表3名・3問のトレースと全パラメータの診断を表示する。
    - 引数: idata1, inference_view, az, mo。戻り値: coords_q, coords_u, summary_a1, summary_b1, summary_theta1。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    coords_q = {"question": ["Q001", "Q002", "Q003"]}
    coords_u = {"user": ["USER0001", "USER0002", "USER0003"]}
    summary_a1 = az.summary(idata1, var_names=["a"], ci_kind="hdi", ci_prob=0.94)
    summary_b1 = az.summary(idata1, var_names=["b"], ci_kind="hdi", ci_prob=0.94)
    summary_theta1 = az.summary(idata1, var_names=["theta"], ci_kind="hdi", ci_prob=0.94)
    mo.vstack([inference_view(idata1, ["a", "b"], coords=coords_q),
               inference_view(idata1, ["theta"], coords=coords_u),
               mo.ui.table(summary_a1.reset_index()), mo.ui.table(summary_b1.reset_index()),
               mo.ui.table(summary_theta1.reset_index())])
    return (coords_q, coords_u, summary_a1, summary_b1, summary_theta1,)


@app.cell
def _(df, idata1, ability_summary, posterior_mean, np, plt, mo):
    """- 名前で能力値を合わせ、同点で異なる解答を比較する。
    - 引数: df, idata1, ability_summary, posterior_mean, np, plt, mo。戻り値: df_sum1, x1_mean, x1_std, df_62_1, selected_users, w1, b_mean1, w3。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    df_sum1, x1_mean, x1_std = ability_summary(df, idata1, ddof=1)
    df_62_1 = df_sum1.loc[np.isclose(df_sum1["素点"], 62)]
    # - 新しい推論結果から選び、古い実行結果の固定IDを混ぜない。
    selected_users = [df_62_1["能力値"].idxmin(), df_62_1["能力値"].idxmax()]
    w1 = df.loc[selected_users]
    b_mean1 = posterior_mean(idata1, "b").sel(question=df.columns.to_list()).values
    w3 = (w1 * b_mean1).sum(axis=1) / w1.sum(axis=1)
    _score, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(df_sum1["偏差値"], df_sum1["能力値"], s=3)
    _ax.set(xlabel="Standardized raw score", ylabel="Standardized ability",
            title="Raw scores and MCMC abilities")
    _score.tight_layout()
    plt.close(_score)
    mo.vstack([mo.ui.table(df_sum1.head(10).rename(columns={
        "素点": "Raw score", "偏差値": "Standardized score", "能力値": "MCMC ability"})),
        _score, mo.md(f"Score 62: {len(df_62_1)} users; selected: {selected_users}"),
        mo.ui.table(w1), mo.ui.table(w3.rename("Mean difficulty of correct answers").reset_index())])
    return (df_sum1, x1_mean, x1_std, df_62_1, selected_users, w1, b_mean1, w3,)


@app.cell
def _(idata1, response, user_idx, question_idx, users, questions, irt_model, pm, sampling_mode, run_inference, mo):
    """- ADVIの学習回数と標本数を保持して推論する。
    - 引数: idata1, response, user_idx, question_idx, users, questions, irt_model, pm, sampling_mode, run_inference, mo。戻り値: model2, mean_field, idata2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    mo.stop(not run_inference.value or "posterior" not in idata1, mo.md("Run MCMC first."))
    model2 = irt_model(response, user_idx, question_idx, users, questions)
    # - MCMC終了後に同一モデルでADVIを実行し、計算資源の競合を避ける。
    with model2:
        # - 元教材にない乱数種を固定し、同一環境の再実行条件を揃える。
        mean_field = pm.fit(n=20000 if sampling_mode.value == "book" else 500,
                           method="advi", random_seed=42, backend="numba",
                           obj_optimizer=pm.adam(), progressbar=False)
        idata2 = mean_field.sample(draws=2000 if sampling_mode.value == "book" else 100, random_seed=42)
    # - ADVI.sampleは変数の保存指定がないため、一時作成した約0.8GBのlogitを解放する。
    del idata2["posterior"]["logit_p"]
    return (model2, mean_field, idata2,)


@app.cell
def _(df, df_sum1, idata1, idata2, mean_field, selected_users, x1_mean, x1_std, ability_summary, np, az, plt, mo):
    """- ADVIの損失と同じ2名の事後分布を比較する。
    - 引数: df, df_sum1, idata1, idata2, mean_field, selected_users, x1_mean, x1_std, ability_summary, np, az, plt, mo。戻り値: df_sum2, x2_mean, x2_std, summary_theta2。
    - 前提: 上流セル実行済み。副作用: 計算・表示。失敗時は元の例外を伝える。
    - 使用例: このセルを実行し、下流の表示を確認する。
    """
    # - 元教材のADVIはNumPy標準偏差ddof=0、MCMCはpandasのddof=1を使う。
    _advi_table, x2_mean, x2_std = ability_summary(df, idata2, ddof=0)
    df_sum2 = df_sum1.assign(能力値2=_advi_table["能力値"])
    summary_theta2 = az.summary(idata2, var_names=["theta"], kind="stats", ci_kind="hdi", ci_prob=0.94)
    _compare, _axes = plt.subplots(1, 3, figsize=(14, 4))
    _axes[0].plot(mean_field.hist)
    _axes[0].set(title="ADVI loss", xlabel="Iteration", ylabel="Negative ELBO")
    # - 同じ受験者を名前で選び、各手法の全体スケールで事後標本を変換する。
    for _ax, _title, _idata, _center, _scale in [
        (_axes[1], "MCMC", idata1, x1_mean, x1_std),
        (_axes[2], "ADVI", idata2, x2_mean, x2_std),
    ]:
        _theta = _idata["posterior"]["theta"].sel(user=selected_users)
        _samples = _theta.transpose("chain", "draw", "user").values.reshape(-1, 2)
        _ax.boxplot(50 + 10 * (_samples - _center) / _scale, tick_labels=selected_users)
        _ax.set(title=_title, ylabel="Standardized ability")
    _compare.tight_layout()
    plt.close(_compare)
    mo.vstack([mo.md("ADVI draws are approximate posterior samples, not MCMC chains."),
        _compare, mo.ui.table(df_sum2.head(10).rename(columns={"素点": "Raw score",
            "偏差値": "Standardized score", "能力値": "MCMC ability", "能力値2": "ADVI ability"})),
        mo.ui.table(summary_theta2.head(10).reset_index())])
    return (df_sum2, x2_mean, x2_std, summary_theta2,)


# - 直接起動時もmarimoの依存順序で実行する。
if __name__ == "__main__":
    # - Windowsで2つのサンプラーワーカーを起動できる入口を用意する。
    import multiprocessing
    multiprocessing.freeze_support()
    app.run()
