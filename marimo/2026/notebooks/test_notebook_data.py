# - 作成日: 2026-09-27
# - 目的: 固定教材データの内容・型・未知入力の拒否を確認する。
# - 役割: Notebook専用のオフライン回帰検査。
# - 使用: uv run --locked --group notebook python notebooks/test_notebook_data.py
# - 制約: 初回データ取得済み。読取のみ、通信なし。
# - 非対応: 推論・画面操作・EXE配布。
from notebook_data import load_data


def test_notebook_data() -> None:
    """3教材を検査する。引数・戻り値なし。前提はデータ準備済み。
    副作用は英語の検査結果出力。不一致はAssertionError。例: test_notebook_data()。
    """
    print("Checking pinned datasets and rejected input")
    # - 失敗を成功と混同せず、理由を維持して伝える。
    try:
        iris = load_data("iris.csv")
        assert iris["species"].value_counts().tolist() == [50, 50, 50]
        assert abs(iris.loc[iris["species"] == "setosa", "sepal_length"].mean() - 5.006) < 1e-12
        irt = load_data("irt-sample.csv")
        assert irt.shape == (1000, 50)
        scores = load_data("test_scores.csv")
        assert scores.shape == (207, 11)
        assert len(scores.dropna()) == 101
        # - ディレクトリ遡行を含む入力は、読込前に拒否される必要がある。
        try:
            load_data("../iris.csv")
        # - 想定する拒否だけを正常な検査結果とする。
        except ValueError:
            pass
        # - 例外なしは入力制限の退行である。
        else:
            raise AssertionError("Unknown input was accepted")
    # - 検査失敗は元の例外と非ゼロ終了を維持する。
    except Exception:
        print("Failed: dataset checks\n")
        raise
    print("Passed: pinned data, schemas and rejected input\n")


# - 通常起動だけで検査し、import時には実行しない。
if __name__ == "__main__":
    test_notebook_data()
