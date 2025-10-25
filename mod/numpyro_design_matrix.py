# ---- NumPyroを使った計画行列ベイズ推定のモジュール ----

# ---- import ----
# DataFrame
import polars as pl
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random

# ベイズ推定
import numpyro
#from numpyro import render_model, plate
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

# Plot
import arviz as az

# 型アノテーション
from typing import List


def make_jax_design_matrix( df, target_col: str, cat_cols: List[str], num_cols: List[str] ):
    """
    NumPyroのベイズ統計モデルに渡すための計画行列や目的変数を作成する

    Parameters
    ----------
    df : Polars DataFrame
        DataFrame
    target_col : str
        目的変数の列名
    cat_cols: list(str)
        カテゴリ変数の列名のリスト
    num_cols: list(str)
        数値型変数の列名のリスト

    Returns
    -------
    X: jnp.array
        計画行列
    y: jnp.array
        目的変数
    feature_cols: list(str)
        特徴量の列名
    """
    # 計画行列、特徴量の条件確認
    assert (len(cat_cols) > 0) | (len(num_cols) > 0), "特徴量が未選択です。"
    assert target_col in df.columns, "目的変数がDataFrameに含まれていません。"
    # 欠損値行を削除する
    hold_cols = [target_col]
    if (len(cat_cols) > 0):
        hold_cols += cat_cols
    if (len(num_cols) > 0):
        hold_cols += num_cols
    df_drop = df[hold_cols].drop_nulls()
    # カテゴリ変数をダミー変数化する
    if (len(cat_cols) > 0):
        df_drop = df_drop.to_dummies(columns = cat_cols, drop_first = True)
        if (len(num_cols) > 0):
            dummy_cols = [c for c in df_drop.columns if c not in num_cols + [target_col]]
        else:
            dummy_cols = [c for c in df_drop.columns if c not in [target_col]]
    # 数値型変数を標準化する
    if (len(num_cols) > 0):
        list_expr = list()
        for c in num_cols:
            list_expr.append( ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c) )
        df_drop = df_drop.with_columns(list_expr)
    # 計画行列を作成する
    feature_cols = list()
    if (len(cat_cols) > 0):
        feature_cols += dummy_cols
    if (len(num_cols) > 0):
        feature_cols += num_cols
    X = jnp.array(df_drop.select(feature_cols).to_numpy(), dtype = jnp.float32)
    # 目的変数を設定する
    y = jnp.array(df_drop.get_column(target_col).to_numpy(), dtype = jnp.float32)
    # 計画行列, 目的変数, 特徴量の列名を返す
    return X, y, feature_cols


def compute_posterior_predictive_distribution(model, feature_cols, mcmc, N, **model_args):
    """
    モデルの事後予測分布を計算する

    Parameters
    ----------
    model : NumPyro model
        ベイズ統計モデル
    feature_cols: list(str)
        計画行列の特徴量の列名
    mcmc: MCMC
        MCMCサンプル
    N: uint
        計画行列のデータ数
    model_args: any variable
        モデル特有の変数。
        特徴量X など
    
    Returns
    -------
    idata: an InferenceData object
        ベイズ推定データ
    """
    # 予測分布のインスタンス
    posterior_samples = mcmc.get_samples()
    pred = Predictive(model, posterior_samples = posterior_samples)
    ppc = pred(random.PRNGKey(1), **model_args)
    # 予測分布のサンプリング
    coords = {"obs_id": np.arange(N), "coef": feature_cols}
    dims = {"obs": ["obs_id"], "beta": ["coef"], "mu": ["obs_id"]}
    idata = az.from_numpyro(
        mcmc,
        posterior_predictive = ppc,
        coords = coords,
        dims = dims,
    )
    return idata


def compute_posterior_predictive_mean_hdi(idata, var_name = "y", hdi_prob = 0.95):
    """
    事後予測平均と信頼区間(HDI)を計算する

    Parameters
    ----------
    idata: arviz.InferenceData
        事後予測分布
    var_name : str
        ベイズ統計モデルで定義した目的変数の変数名
    hdi_prob: float
        HDI信頼区間の範囲

    Returns
    -------
    y_ppc_mean: np.array
        事後予測平均
    y_ppc_low: np.array
        事後予測HDIの下限
    y_ppc_high: np.array
        後予測HDIの上限
    """
    # 事後予測平均
    y_ppc_mean = np.array( idata["posterior_predictive"][var_name].mean(dim = ("chain", "draw")) )
    # 事後予測のHDI
    hdi_ppc = az.hdi(
        idata,                          # InferenceData 全体を渡す
        group = "posterior_predictive", # 計算対象グループを指定
        var_names = [var_name],         # 対象変数
        hdi_prob = hdi_prob,            # 区間幅
    )
    y_ppc_low = np.array( hdi_ppc[var_name][:, 0] )
    y_ppc_high = np.array( hdi_ppc[var_name][:, 1] )
    return y_ppc_mean, y_ppc_low, y_ppc_high

