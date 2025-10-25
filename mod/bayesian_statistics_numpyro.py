# ---- NumPyroを使ったベイズ統計のモジュール ----

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

# ---- 汎用的なモジュール ----
# ---- モデルの可視化・サンプリング ----
def try_render_model(model, render_name : str, **model_args):
    """
    ベイズ統計モデルを可視化する

    Parameters
    ----------
    model: NumPyro model
        ベイズ統計モデル
    render_name : str
        可視化したモデルのファイル名
    model_args: Any args
        モデルの引数
    """
    try:
        # 確率モデルを作成する
        g = numpyro.render_model(
            model = model,
            model_args = (),
            model_kwargs = model_args,
            render_distributions = True,
            render_params = True
        )
        # 確率モデルの画像を保存する
        outpath = f"{render_name}.svg"
        g.render(render_name, format = "svg", cleanup = True)
        print(f"Model graph saved to: {outpath}")
        # Jupyter なら表示、スクリプトならファイル出力のみ
        try:
            from IPython.display import display, SVG
            display(SVG(filename=outpath))
        except Exception:
            print("Model graph saved to hier_model.svg")
    except Exception as e:
        print(f"(Skip model rendering for {render_name}: {e})")


def run_mcmc(model, num_chains = 4, num_warmup = 1000, num_samples = 1000, thinning = 1, seed = 42, **model_args):
    """
    NumPyroのベイズ統計モデルのサンプリングを実行する。

    Parameters
    ----------
    model : NumPyro model
        ベイズ統計モデル
    num_chains : uint
        サンプリングのチェーン数
    num_warmup : uint
        推定に使用しないMCMCサンプル数
    num_samples :uint
        推定に使用するMCMCサンプル数
    thinning : uint
        num_samples のMCMCサンプルをさらに間引く係数。
    seed : uint
        乱数シード
    model_args: any variable
        モデル特有の変数。
        データ数 N や特徴量X, 目的変数y など

    Returns
    -------
    mcmc : MCMC
        MCMCインスタンス
    """
    sampler = NUTS(model)
    num_devices = jax.local_device_count()
    chain_method = "parallel" if num_devices >= num_chains else "sequential"
    mcmc = MCMC(
        sampler = sampler,
        num_warmup = num_warmup,
        num_samples = num_samples,
        num_chains = num_chains,
        thinning = thinning,
        chain_method = chain_method,
        progress_bar = True,
    )
    mcmc.run(random.PRNGKey(seed), **model_args)
    return mcmc


# ---- 計画行列を使用する場合のモジュール ----
# ---- 前処理 ----
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


# ---- 事後予測分布を計算する ----
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

