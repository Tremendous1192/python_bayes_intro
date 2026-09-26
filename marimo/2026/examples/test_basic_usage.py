# 作成日: 2026-09-26
# 目的: ガイドの平均・表・図・ボタン停止条件を、解析的な期待値と照合する。
# 役割: Notebook専用の回帰確認。各ケースを新規プロセスで実行する。
# 使用方法: uv run --locked --group notebook python examples/test_basic_usage.py
# 制約: env.cmd適用済み、CPython 3.14.7。画面操作は値の差替えによる模擬。
# 非対応: ブラウザー・VS Codeの手動操作、書籍の推論、EXE配布。
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from basic_usage import app


def test_basic_usage(sample_size: int = 10, pressed: bool = False) -> None:
    """平均・表示内容・停止を照合する。

    引数: sample_sizeは10または20、pressedはボタン押下相当の真偽値。
    戻り値: なし。前提: 設定済みのNotebook専用環境。
    副作用: セル実行、描画、英語の検証結果出力。外部データ取得なし。
    例外: 不一致はAssertionError、セルの実行失敗は元の例外を再送出する。
    例: test_basic_usage(20, True)で平均10.5と押下後の結果を確認する。
    """
    print(f"Checking sample size {sample_size}, button pressed {pressed}", flush=True)
    # 失敗を英語で明示して再送出し、検証失敗を正常終了にしない。
    try:
        # 初期ケースは実際のUIを作成し、変更ケースだけセルの定義を差し替える。
        overrides = {}
        # スライダー変更を模擬し、依存するセルが新しい値を使うことを確認する。
        if sample_size != 10:
            overrides["sample_size"] = SimpleNamespace(value=sample_size)
        # ボタンを押したケースだけ、停止条件を通過する定義に差し替える。
        if pressed:
            overrides["run_calculation"] = SimpleNamespace(value=True)
        _, definitions = app.run(defs=overrides)

        # NumPyの平均を再利用せず、等差数列の解析解で期待値を決める。
        expected_mean = (sample_size + 1) / 2
        # 期待する観測値を整数列から作り、型と内容を別々に確認する。
        expected_values = [float(value) for value in range(1, sample_size + 1)]
        assert definitions["sample_mean"] == expected_mean
        assert str(definitions["observations"].dtype) == "float64"
        assert definitions["observations"].tolist() == expected_values
        assert definitions["sample_table"].data == {"Value": expected_values}
        assert list(definitions["axis"].lines[0].get_ydata()) == expected_values
        assert list(definitions["axis"].lines[1].get_ydata()) == [expected_mean] * 2
        definitions["figure"].canvas.draw()
        assert definitions["plt"].get_fignums() == []

        # 押下時だけ結果を定義し、未押下時は停止メッセージで終わることを確認する。
        if pressed:
            assert definitions["confirmed_mean"] == expected_mean
        # 未押下でも計算結果が残る退行を検出する。
        else:
            assert "confirmed_mean" not in definitions
    # セルまたは期待値の照合が失敗した場合は、理由を保持して呼出元へ伝える。
    except Exception:
        print("Failed: notebook regression check\n", flush=True)
        raise
    print("Passed: mean, table, plot, and button gate\n", flush=True)


# 通常起動では3ケースを順番に別プロセスで実行し、前の状態を引き継がない。
if __name__ == "__main__":
    # ケースは初期状態、入力変更後の未押下、入力変更後の押下相当の3種類。
    cases = {"default": (10, False), "changed": (20, False), "pressed": (20, True)}
    # 子プロセスは指定された1ケースだけ実行する。
    if len(sys.argv) == 2:
        test_basic_usage(*cases[sys.argv[1]])
    # 引数なしの入口は、同じPythonで期限付き・逐次実行する。
    else:
        # 同時に複数の描画・計算プロセスを立ち上げない。
        for case in cases:
            subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), case],
                check=True,
                timeout=90,
            )
