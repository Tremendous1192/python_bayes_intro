# - 作成日: 2026-09-27
# - 目的: 3クラス教材の3種類のモデルで観測・事前分布の対応を保持する。
# - 役割: Notebook専用のモデル定義。重複によるカテゴリ番号の不一致を防ぐ。
# - 使用: three_class_model(X)、またはscale_prior・orderedで教材の比較例を指定する。
# - 制約: 有限な1次元観測、3クラス、CPU。推論は呼出側が担当。
# - 非対応: 任意クラス数・EXE・GPU。
import numpy as np
import pymc as pm


def three_class_model(
    X: np.ndarray, *, scale_prior: bool = False, ordered: bool = False,
) -> pm.Model:
    """- カテゴリカル潜在変数のモデルを作る。
    - 引数: Xは花弁幅、scale_priorは標準偏差の事前分布、orderedは平均の順序制約。
    - 戻り値: PyMCモデル。前提: 空でない有限の1次元配列。
    - 副作用: モデル内への変数登録。失敗: 不正な観測はValueError。
    - 使用例: three_class_model(np.array([0.2, 1.0, 2.0]), ordered=True)。
    """
    # - 入力形状や非有限値を推論前に拒否する。
    if X.ndim != 1 or not X.size or not np.isfinite(X).all():
        raise ValueError("Expected a nonempty finite one-dimensional array.")
    # - 各比較例に専用のモデルを作り、共有状態を避ける。
    with pm.Model() as model:
        X_data = pm.Data("X_data", X)
        p = pm.Dirichlet("p", a=np.ones(3))
        s = pm.Categorical("s", p=p, shape=X.shape)
        # - 正の差を2段使い、mu0 < mu1 < mu2を保証する。
        if ordered:
            mu0 = pm.HalfNormal("mu0", sigma=10.0)
            delta0 = pm.HalfNormal("delta0", sigma=10.0)
            mu1 = pm.Deterministic("mu1", mu0 + delta0)
            delta1 = pm.HalfNormal("delta1", sigma=10.0)
            mu2 = pm.Deterministic("mu2", mu1 + delta1)
            mus = pm.Deterministic("mus", pm.math.stack([mu0, mu1, mu2]))
        # - 制約なしでは成分番号の交換を許し、教材の問題を観察する。
        else:
            mus = pm.Normal("mus", mu=0.0, sigma=10.0, shape=3)
        mu = pm.Deterministic("mu", mus[s])
        # - 標準偏差へ直接事前分布を置く比較例を残す。
        if scale_prior:
            sigmas = pm.HalfNormal("sigmas", sigma=10.0, shape=3)
            sigma = pm.Deterministic("sigma", sigmas[s])
            pm.Normal("X_obs", mu=mu, sigma=sigma, observed=X_data)
        # - 精度モデルの標準偏差は決定論的に計算する。
        else:
            taus = pm.HalfNormal("taus", sigma=10.0, shape=3)
            pm.Deterministic("sigmas", 1 / pm.math.sqrt(taus))
            tau = pm.Deterministic("tau", taus[s])
            pm.Normal("X_obs", mu=mu, tau=tau, observed=X_data)
    return model
