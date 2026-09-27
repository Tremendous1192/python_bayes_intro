# - 作成日: 2026-09-27
# - 目的: IRTのCPUバックエンド変更を対数密度・勾配の一致と実測で検証する。
# - 役割: Notebook専用。Numbaの結果を非JITのC実装と独立した数式で照合する。
# - 使用: uv run --locked --group notebook python notebooks/test_irt_backend.py
# - 制約: 全50000回答、float64、Numba上限2スレッド、生成物は.cache内。
# - 非対応: EXE・GPU・別機種への速度保証・短縮標本による収束保証。
import json
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
import pandas as pd
from pytensor.compile.mode import Mode, get_mode
from scipy import special, stats

from irt_model import irt_model
from notebook_data import load_data


def test_irt_backends() -> None:
    """- CとNumbaの密度・勾配を3つの固定点で比較する。
    - 引数・戻り値: なし。前提: 固定環境、登録済みキャッシュ先。
    - 副作用: CPUコンパイル・英語出力・.cacheへの測定記録。
    - 失敗: API不適合や数値不一致を伝える。使用例: test_irt_backends()。
    """
    print("Checking IRT C and Numba backends on all 50000 responses", flush=True)
    # - コンパイル失敗や数値差を隠して速度だけを比較しない。
    try:
        assert numba.get_num_threads() <= 2
        df = load_data("irt-sample.csv")
        table = df.rename_axis("user").reset_index().melt(
            id_vars="user", var_name="question", value_name="response")
        user_idx, users = pd.factorize(table["user"], sort=False)
        question_idx, questions = pd.factorize(table["question"], sort=False)
        response = table["response"].to_numpy()
        model = irt_model(response, user_idx, question_idx, users, questions)
        point = {"theta": np.linspace(-1.0, 1.0, len(users)),
                 "a_log__": np.linspace(-0.2, 0.2, len(questions)),
                 "b": np.linspace(-0.5, 0.5, len(questions))}
        results, compiled = {}, {}
        # - 初回のJITを含む時間と、その後の反復時間を別に記録する。
        for backend in ("c", "numba"):
            started = perf_counter()
            # - Modelのcompile系はsampleのbackendではなくPyTensorのmodeを受け取る。
            # - この環境のFAST_RUNはNumbaなので、非JIT側はC VMを明示する。
            mode = Mode(linker="cvm", optimizer="fast_run") if backend == "c" else get_mode("NUMBA")
            print(f"{backend} linker: {type(mode.linker).__name__}", flush=True)
            logp = model.compile_logp(mode=mode)
            gradient = model.compile_dlogp(mode=mode)
            logp(point)
            gradient(point)
            first_seconds = perf_counter() - started
            started = perf_counter()
            # - 同じ入力で5回評価し、温まった密度・勾配の合計時間を測る。
            for _ in range(5):
                logp(point)
                gradient(point)
            warm_seconds = (perf_counter() - started) / 5
            compiled[backend] = (logp, gradient)
            results[backend] = {"compile_and_first_seconds": first_seconds,
                                "warm_pair_seconds": warm_seconds}
            print(f"{backend}: first={first_seconds:.3f}s; warm pair={warm_seconds:.6f}s", flush=True)
        # - 位置を変えても、C・Numba・独立したSciPyの密度が一致することを確認する。
        for shift in (-0.2, 0.0, 0.2):
            current = {name: values + shift for name, values in point.items()}
            theta, a, b = current["theta"], np.exp(current["a_log__"]), current["b"]
            logits = a[question_idx] * (theta[user_idx] - b[question_idx])
            expected = np.where(response == 1, special.log_expit(logits),
                                special.log_expit(-logits)).sum()
            expected += stats.norm.logpdf(theta).sum() + stats.halfnorm.logpdf(a).sum()
            expected += stats.norm.logpdf(b).sum() + current["a_log__"].sum()
            c_logp, c_grad = compiled["c"]
            n_logp, n_grad = compiled["numba"]
            np.testing.assert_allclose(c_logp(current), expected, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(n_logp(current), expected, rtol=1e-10, atol=1e-7)
            np.testing.assert_allclose(n_grad(current), c_grad(current), rtol=1e-9, atol=1e-7)
        target = Path(__file__).resolve().parents[1] / ".cache" / "validation"
        target.mkdir(exist_ok=True)
        results["numba_threads"] = numba.get_num_threads()
        (target / "irt-backends.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    # - 失敗の詳細を保持して終了する。
    except Exception:
        print("Failed: IRT backend comparison\n", flush=True)
        raise
    print("Passed: IRT backend comparison\n", flush=True)


# - 直接実行時だけコンパイルと数値照合を行う。
if __name__ == "__main__":
    test_irt_backends()
