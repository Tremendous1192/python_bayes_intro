# - 作成日: 2026-09-28。教材CSVの検査と、明示指定時だけの不足分取得を行う。
# - 使用: プロジェクトのPythonで python data/prepare_data.py [--download] を実行する。
# - 制約: Notebook用の固定3ファイルのみ。既存CSVは上書きせず、依存の導入は行わない。
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

# - スクリプトと同じdataフォルダを使い、呼出元の作業ディレクトリに依存しない。
DATA_ROOT = Path(__file__).resolve().parent


def prepare(download: bool = False) -> None:
    """固定CSVを検査し、許可時だけ不足分を取得する。

    sources.jsonのURL・SHA-256を使用する。不一致・取得失敗は例外とし、
    既存ファイルは置換しない。既存CSVのCRLFだけは読込時にLFへ戻して照合する。
    """
    sources = json.loads((DATA_ROOT / "sources.json").read_text(encoding="utf-8"))
    # - 取得対象を3本に限定し、JSONから任意パスを指定させない。
    for name in ("iris.csv", "test_scores.csv", "irt-sample.csv"):
        target = DATA_ROOT / name
        source = sources[name]
        # - 既存データは取得許可の有無によらず、読み取りだけで検査する。
        if target.exists():
            data = target.read_bytes()
        elif download:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "bayes-notebook"})
            # - 接続・読取時間と最大サイズを制限する。
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(2_000_001)
            # - 保存前は元バイト列を厳密に照合し、想定外の応答を残さない。
            if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Invalid download: {name}")
            # - 排他作成により、同時処理や利用者のファイルを上書きしない。
            with target.open("xb") as handle:
                handle.write(data)
        else:
            raise FileNotFoundError(
                f"Missing {name}; run uv run --locked --offline --group notebook "
                "python data/prepare_data.py --download from the project root."
            )
        # - GitのCRLF変換だけをメモリ上で戻し、固定ハッシュやファイルは変更しない。
        if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != source["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}; preserve the file and inspect it.")
        print(f"Passed: {name}, {len(data)} bytes")


def main() -> int:
    """引数を検査してデータを準備し、成功は0、検査・取得失敗は1を返す。"""
    # - 未知の引数や省略形では取得せず、argparseの終了コード2で停止する。
    parser = argparse.ArgumentParser(description="Check pinned teaching datasets.", allow_abbrev=False)
    parser.add_argument("--download", action="store_true", help="Download missing CSV files only.")
    args = parser.parse_args()
    try:
        prepare(download=args.download)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
