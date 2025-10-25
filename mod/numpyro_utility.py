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
    NumPyroのベイズ統計モデルでサンプリングする

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
