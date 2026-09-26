# 目的
2026年9月26日時点の開発環境で `Pythonでスラスラわかる ベイズ推論「超」入門` のコードを書きなおす。

# 初回の環境構築
`VS Code` + `uv` + `marimo` で開発する。
## 1. インストーラーで開発用ツールをインストールする
1. Visual Studio Code
    * `https://code.visualstudio.com/download`
1. Miniforge(Python)
    * `https://github.com/conda-forge/miniforge`
    * Download and execute the Windows installer. の文をクリックしてインストーラーをダウンロードします
    * インストール先フォルダ `C:\Miniforge3`
    * システム環境変数のPATH `C:\Miniforge3\Scripts`
1. Git
    * `https://gitforwindows.org/`
    * Version `2.54.0`
1. Graphviz
    * `https://graphviz.org/download/`
    * 64 bit exe
    * `Add Graphviz to the system PATH for all users`
    * ターミナルで`dot -v`を入力するとインストール成功を確認できる
1. MinGW
    * https://github.com/niXman/mingw-builds-binaries
    * 記載していない内容はデフォルト値を選択する。
    * バージョン番号はデフォルト値(最新)を選択する
    * (必須)**64bit** を選択すること。
    * OSはデフォルト値(`win32`)を選択する
    * リビジョンはデフォルト値(最新)を選択する
    * ランタイムはデフォルト値(`msvcrt`)を選択する
    * (重要)`Install in` のパスをCドライブ直下とする
        * `C:/`
    * (必須)`システム環境変数の編集`の`PATH`に書きのパスを追加する
        * `C:\mingw64\bin`
1. MinGWをWindows defenderの例外フォルダに設定する。
    * 参考 https://starfort.cocolog-nifty.com/voorlihter/2024/05/post-8905fe.html
    * `C:\mingw64` を除外する
1. 開発環境を有効にするために、PCを再起動する

## 2. VS Codeの拡張機能をインストールする
1. `Python`
1. `marimo`

## 3. VS Code での準備
1. VS Codeの`Command Palette`で`Terminal: Select Default Profile`を選び,`Command Prompt`を選択する。
    * 理由は忘れたがPower Shellのままだと不都合があった。
1. VS CodeのTerminalで`conda`を実行する。
1. VS CodeのTerminalで`conda init`を実行する。
1. VS CodeのTerminalで`uv`をインストールする。
    * `winget install --id=astral-sh.uv -e`
    * VS CodeのTerminalを閉じて、もう一度開くと使用可能になる


