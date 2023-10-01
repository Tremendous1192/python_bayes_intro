# 書籍『Pythonでスラスラわかりベイズ推論「超」入門』サポートサイト

<div align="left">
<img src="images/表紙-v1.png" width="300">
</div>
　当サイトは 、書籍『Pythonでスラスラわかりベイズ推論「超」入門』のサポートサイトです。<br> 　本書は、ベイズ推論ライブラリ「PyMC」で、 ベイズ推論プログラミングができるようになる本です。<br>　ベイズ推論理解のためには、確率分布などの数学知識が必須です。本書であれば、数学知識が不十分な読者も、Python実習を通じて簡単にベイス推論が理解できます。




## 実習Notebookリンク
　本書の実習コードは、Google Colabで動かすことを前提に、すべてGithub(当サポートサイト)で公開しています。  

[実習Notebook一覧](https://github.com/makaishi2/python_bayes_intro/tree/main/notebooks)

[実習Notebookの動かし方](refs/how-to-run.md)



<!---

## Amazonへのリンク

[単行本](https://www.amazon.co.jp/dp/4296110322) 

[Kindle](https://www.amazon.co.jp/dp/B09G622WB6/)  

-->



## 本書の特徴

* ベイズ推論でモデルを構築する場合に必須である確率分布の初歩を、プログラミングモデルと対比しながら理解できるようになります

* PyMCとArVizの使い方を一歩一歩学べます

* 「くじ引きを5回引いた結果からくじの当たる確率を類推する」という簡単な例題を通して、ベイズ推論の考え方を理解できます

* 「正規分布の平均、標準偏差を推論する」というシンプルな問題から「潜在変数モデル」という高度な問題まで様々なベイズ推論の仕組みを実習プログラムを通じて理解できます

* ABテストや線形回帰モデルの効果検証など、業務観点でのベイズ推論利用パターンを学ぶことができます

* 各章・節の最後のコラムで、「事前分布と事後分布の違い」、「HDIの意味」や、「target_acceptパラメータによるチューニング」「変分推論法の利用」など、知っておくと役にたつ、やや高度な概念や手法を理解できます



## 主な想定読者



　本書では、scikit-learnなどのライブラリを利用する**普通の機械学習はマスターした上で**、**次のステップでベイズ推論を学習したい**という読者の方を想定しています。<br>プログラミングとの類推を活用して確率などの数学を説明する部分もあるので、**ある程度のPythonプログラミングスキルは前提**としています。具体的な知識レベルは以下のとおりです。



* Python文法の基礎知識

  - 整数型、浮動小数点数型、ブーリアン型などの基本型

  - 関数呼び出しにおけるオプション付き引数

  - オブジェクト指向プログラミングの基礎概念(クラス、インスタンス、コンストラクタ)


* NumPy, pandas, matplotlib, Seabornの基本的な操作



　数学に関しては、極力、**高校１年程度の数学知識で読み進めることができる**よう心がけました。確率分布の説明などで、数式が出てくる箇所もありますが、数式をスルーしても先に読み進められるよう工夫したつもりです。逆に Pythonコードと数学概念との対応はとても重視しているので、読者の方は極力、本書の前提である**Google Colabで実習コードをを動かしながら本書をを読み進めていただく**ことを推奨します。




## 目次

[目次リンク](refs/目次.md)







## その他解説記事
|ソース  |タイトルとリンク  |補足|
|---|---|---|
|当サポートサイト|[3クラス潜在変数モデル](refs/3クラス潜在変数モデル.pdf)|5.4節の潜在変数モデルは対象を3クラスに拡張可能です。その解説をしています|



## リンク集

### 著者発信の情報

|ソース  |タイトルとリンク  |補足|
|---|---|---|



### 外部リンク


|ソース  |タイトルとリンク  |補足|
|---|---|---|

***


## 正誤訂正・FAQ

<!---
* [Notebook補足情報](notebook-ref.md)
-->  

* [正誤訂正](refs/errors.md)

* [FAQ](refs/faqs.md)

