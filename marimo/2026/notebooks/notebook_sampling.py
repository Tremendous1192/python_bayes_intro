# - 作成日: 2026-09-27
# - 目的: 各章のMCMC条件を明示し、Windowsで計算資源を制限する。
# - 役割: 重複するサンプラー選択・上限処理の不一致を防ぐNotebook専用処理。
# - 使用: sample_model(model, mode="book")。ボタン停止判定の後に呼ぶ。
# - 制約: 既定Cは1プロセス。IRTは同時2チェーン、各BLAS1・Numba上限2。
# - 非対応: GPU、任意サンプラー、EXE配布。quickは収束評価には使わない。
import pymc as pm


def sample_model(model, mode: str = "book", *, chains: int = 4,
                 draws: int = 1000, tune: int = 1000,
                 target_accept: float = 0.8, seed: int = 42,
                 var_names=None, backend: str = "c"):
    """条件を明示してMCMCを行う。引数はモデル・モード・チェーン数・反復数・受容率・種・保存変数・CPUバックエンド。
    戻り値はDataTree。前提: 固定環境とg++。副作用はCPU計算とプロジェクト内コンパイル。
    例外はPyMCから伝搬。例: sample_model(model, "book", chains=1, draws=2000)。
    """
    # - 未知のモードで反復数を黙って変えない。
    if mode not in {"book", "quick"}:
        raise ValueError("Expected book or quick sampling mode.")
    # - 実測で必要となったIRTのNumbaと、既存教材のCに対応を限定する。
    if backend not in {"c", "numba"}:
        raise ValueError("Expected c or numba backend.")
    # - 専用Notebook環境のスレッド上限を確認し、CPUの過剰使用を避ける。
    if backend == "numba":
        import numba
        # - 起動設定のない別環境では、暗黙に全CPUを使用せず停止する。
        if numba.get_num_threads() > 2:
            raise RuntimeError("Numba requires at most 2 threads in the registered Notebook environment.")
    # - 短縮検査は配線確認専用で、チェーンの教材上の意味は維持する。
    if mode == "quick":
        draws, tune = min(draws, 100), min(tune, 150)
    # - 重いIRTだけ同時2チェーンとし、標本生成中はNumba合計最大4スレッドに制限する。
    cores = 2 if backend == "numba" else 1
    # - 離散潜在変数にも対応するPyMCを固定し、自動nutpie選択を避ける。
    with model:
        # - Windowsのspawnを明示し、BLASは各子プロセス1スレッドへ割り当てる。
        return pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                         blas_cores=cores, mp_ctx="spawn" if cores > 1 else None,
                         random_seed=seed, target_accept=target_accept,
                         nuts_sampler="pymc", backend=backend, progressbar=False,
                         var_names=var_names)


def posterior_mean(idata, name: str):
    """チェーンとdrawを平均する。引数はDataTreeと変数名、戻り値はDataArray。
    前提: posteriorが存在する。副作用なし。未知変数はKeyError。
    例: posterior_mean(idata, "mu")。座標を保持して表の行順との混同を防ぐ。
    """
    return idata["posterior"][name].mean(dim=("chain", "draw"))
