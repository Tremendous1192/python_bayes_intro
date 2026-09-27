# - 作成日: 2026-09-27
# - 目的: MCMCとADVIでIRTの同一モデル・座標・能力値変換を使う。
# - 役割: Notebook専用。2つの推論方式で受験者の並びがずれる不具合を防ぐ。
# - 使用: irt_model(response, user_idx, question_idx, users, questions)。
# - 制約: 2PL、完全な0/1観測、CPU。計算・保存の開始は呼出側で制御する。
# - 非対応: 欠損回答・GPU・EXE。
import numpy as np
import pandas as pd
import pymc as pm


def irt_model(response, user_idx, question_idx, users, questions) -> pm.Model:
    """- 同じ受験者・設問座標で2PLモデルを構築する。
    - 引数: 回答値・受験者番号・設問番号の1次元配列と対応する名前。
    - 戻り値: PyMCモデル。前提: 番号は範囲内、各配列は同じ長さ。
    - 副作用: モデル内変数の登録。失敗: 不正な観測はValueError。
    - 使用例: irt_model(np.array([1]), np.array([0]), np.array([0]), ["U1"], ["Q1"])。
    """
    # - 不一致の配列をbroadcastで誤って解釈しない。
    if not (response.shape == user_idx.shape == question_idx.shape) or response.ndim != 1:
        raise ValueError("Response and index arrays must have identical one-dimensional shapes.")
    # - 空配列、欠損、範囲外ラベルを推論前に拒否する。
    if (not response.size or not np.isin(response, [0, 1]).all()
            or user_idx.min() < 0 or user_idx.max() >= len(users)
            or question_idx.min() < 0 or question_idx.max() >= len(questions)):
        raise ValueError("Invalid binary responses or coordinate indices.")
    # - 2つの推論方式で事前分布と観測の登録を一致させる。
    with pm.Model(coords={"user": users, "question": questions}) as model:
        response_data = pm.Data("response_data", response)
        theta = pm.Normal("theta", mu=0.0, sigma=1.0, dims="user")
        a = pm.HalfNormal("a", sigma=1.0, dims="question")
        b = pm.Normal("b", mu=0.0, sigma=1.0, dims="question")
        logit_p = pm.Deterministic("logit_p", a[question_idx] * (theta[user_idx] - b[question_idx]))
        pm.Bernoulli("obs", logit_p=logit_p, observed=response_data)
    return model


def ability_summary(df: pd.DataFrame, idata, *, ddof: int = 1):
    """- 受験者名で揃えた素点・偏差値・能力値を返す。
    - 引数: dfは回答表、idataは事後分布、ddofは元教材の標準偏差条件。
    - 戻り値: 集計表、能力平均、能力標準偏差。前提: 正しいuser座標。
    - 副作用: なし。失敗: 座標不一致はKeyError、変動なしはValueError。
    - 使用例: ability_summary(df, idata1, ddof=1)。
    """
    raw_score = df.mean(axis=1) * 100
    ability = idata["posterior"]["theta"].mean(("chain", "draw")).sel(user=df.index.to_list())
    center, scale = float(ability.mean()), float(ability.std(ddof=ddof))
    # - 定数標本では偏差値の定義が成立しないため、ゼロ除算を避ける。
    if scale <= 0 or raw_score.std(ddof=1) <= 0:
        raise ValueError("Standardized scores require nonzero variance.")
    table = pd.DataFrame({"素点": raw_score,
                          "偏差値": 50 + 10 * (raw_score - raw_score.mean()) / raw_score.std(ddof=1),
                          "能力値": 50 + 10 * (ability.values - center) / scale}, index=df.index)
    return table, center, scale
