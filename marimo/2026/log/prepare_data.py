# - 作成日: 2026-09-27
# - 目的: 教材データ3本を固定コミットから準備し、改変を検出する。
# - 役割: 明示的な初回取得の入口。Notebookからは呼ばない。
# - 使用: uv run --locked --group notebook python log/prepare_data.py --download
# - 制約: 保存先はプロジェクト直下のdata。各取得30秒・2MB以下。
# - 非対応: 自動更新、既存データの上書き、依存導入、EXE配布。
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

# - logの親を基準にし、呼出元の作業ディレクトリには依存しない。
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def prepare(download: bool = False) -> None:
    """固定ファイルを検査する。引数は不足時の取得許可、戻り値なし。
    前提: sources.jsonが存在する。副作用: 許可時のみ不足CSVを保存する。
    不一致・取得失敗は例外。例: prepare(True)。既存ファイルは置換しない。
    """
    sources = json.loads((DATA_ROOT / "sources.json").read_text(encoding="utf-8"))
    # - 取得対象を3本に限定し、JSONから任意パスを指定させない。
    for name in ("iris.csv", "test_scores.csv", "irt-sample.csv"):
        target = DATA_ROOT / name
        source = sources[name]
        # - 既存データは読み取りのみで検査する。
        if target.exists():
            data = target.read_bytes()
        # - 明示指定した初回だけネットワークへ接続する。
        elif download:
            request = urllib.request.Request(source["url"], headers={"User-Agent": "bayes-notebook"})
            # - 接続・読取時間と最大サイズを制限する。
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(2_000_001)
            # - 想定外の応答や内容は保存しない。
            if len(data) > 2_000_000 or hashlib.sha256(data).hexdigest() != source["sha256"]:
                raise ValueError(f"Invalid download: {name}")
            # - 排他作成により、同時処理や利用者のファイルを上書きしない。
            with target.open("xb") as handle:
                handle.write(data)
        # - 学習中の不足は勝手に取得せず、準備手順を示す。
        else:
            raise FileNotFoundError(f"Missing {name}; run python log/prepare_data.py --download.")
        # - 利用者の変更や破損を検出して停止する。
        if hashlib.sha256(data).hexdigest() != source["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}; preserve the file and inspect it.")
        print(f"Passed: {name}, {len(data)} bytes")


# - 明示的なコマンド起動時だけ取得オプションを解釈する。
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify pinned teaching datasets.")
    parser.add_argument("--download", action="store_true", help="Download missing files only.")
    prepare(parser.parse_args().download)
