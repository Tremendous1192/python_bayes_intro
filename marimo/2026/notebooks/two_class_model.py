# - 作成日: 2026-09-27
# - 目的: 本編・簡略版・FAQで重複する2クラス潜在モデルを一致させる。
# - 役割: Notebook専用のモデル定義。3教材間の事前分布の取り違えを防ぐ。
# - 使用: two_class_model(X, fixed_probability=False)でモデルを作る。
# - 制約: 1次元の有限な観測、2クラス、CPU。推論は呼出側が実施する。
# - 非対応: 任意クラス数・EXE・GPU。
import numpy as np
import pymc as pm


def two_class_model(
    X: np.ndarray, *, fixed_probability: bool = False,
    scale_prior: bool = False, ordered: bool = False,
) -> pm.Model:
    """- 元教材の2クラスモデルを構築する。
    - 引数: Xは花弁幅、fixed_probabilityはp=0.5、scale_priorは標準偏差の事前分布、
      orderedは平均の大小制約。戻り値: PyMCモデル。
    - 前提: Xは空でない有限の1次元配列。副作用: モデル内部への変数登録。
    - 失敗: 不正な観測はValueError。使用例: two_class_model(np.array([1.0, 2.0]))。
    """
    # - 形状や欠損の誤りをサンプリング前に検出する。
    if X.ndim != 1 or not X.size or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite one-dimensional array.")
    # - モデルごとに独立した変数を登録し、Notebook間で状態を共有しない。
    with pm.Model() as model:
        X_data = pm.Data("X_data", X)
        # - 簡略版だけは混合確率を固定し、pの事後サンプルを作らない。
        if fixed_probability:
            p = 0.5
        # - 本編とFAQでは混合確率も推論する。
        else:
            p = pm.Uniform("p", lower=0.0, upper=1.0)
        s = pm.Bernoulli("s", p=p, shape=X.shape)
        # - 正の差を使うことで成分の番号の入れ替わりを防ぐ。
        if ordered:
            mu0 = pm.HalfNormal("mu0", sigma=10.0)
            delta0 = pm.HalfNormal("delta0", sigma=10.0)
            mu1 = pm.Deterministic("mu1", mu0 + delta0)
            mus = pm.Deterministic("mus", pm.math.stack([mu0, mu1]))
        # - 制約なしの教材ではラベルスイッチも結果として残す。
        else:
            mus = pm.Normal("mus", mu=0.0, sigma=10.0, shape=2)
        mu = pm.Deterministic("mu", mus[s])
        # - 標準偏差に直接事前分布を置く比較例を保持する。
        if scale_prior:
            sigmas = pm.HalfNormal("sigmas", sigma=10.0, shape=2)
            sigma = pm.Deterministic("sigma", sigmas[s])
            pm.Normal("X_obs", mu=mu, sigma=sigma, observed=X_data)
        # - 精度の事前分布は標準偏差の事前分布と同一ではない。
        else:
            taus = pm.HalfNormal("taus", sigma=10.0, shape=2)
            pm.Deterministic("sigmas", 1 / pm.math.sqrt(taus))
            tau = pm.Deterministic("tau", taus[s])
            pm.Normal("X_obs", mu=mu, tau=tau, observed=X_data)
    return model
