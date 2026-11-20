# music-to-character
名古屋工業大学のメディア系演習2のプロジェクト用のリポジトリ

楽曲 (WAV / MP3 / FLAC / OGG) を分析し、その印象に合ったキャラクター (SVG) を生成する。
歌声・歌詞は分析対象外 (単旋律の楽器曲を想定)。

## 使い方

```
pip install -r requirements.txt
streamlit run app.py          # 画面を起動
python -m pytest              # テスト
```

## 処理の流れ

```
音声 -> audio/ -> features/ -> impressions/ -> character/ -> SVG
        読込・前処理  特徴量 A〜H   印象 1〜8      パーツ値・描画
```

統合は `musicchar/pipeline.py`、画面は `app.py`。各層は `musicchar/models` 以外に依存しない
(`tests/test_architecture.py` で検査)。

## 設定ファイル (`config/`)

数値や計算式のパラメータはコードに埋め込まず、ここで管理する。**値は暫定**で、実際の楽曲で調整する。

| ファイル | 内容 |
|---|---|
| `audio.yaml` | サンプリングレート、無音トリム、音高推定の範囲 |
| `features.yaml` | メロディ・調・コード・アーティキュレーションの抽出パラメータ |
| `normalization.yaml` | 特徴量 -> 0〜1 の正規化範囲 |
| `impressions.yaml` | 印象 1〜8 の入力特徴量・重み・向き |
| `character_parts.yaml` | パーツごとの使う印象・変化方向・最小値/最大値・合成方式 (mean / weighted_mean) |
| `render.yaml` | SVG のレイアウト・色・エフェクトと装飾の配置 |

## 欠損の扱い

抽出できなかった特徴量は `None` とし、理由を記録する (0 で代用しない)。
印象は利用できる入力だけで重みを再正規化して計算し、入力がすべて欠けた印象は `None`。
`None` の印象に依存するパーツは、利用できる印象だけ (なければ中立値 0.5) で決め、画面に「推定不能」と表示する。

## 既知の制約

- 音高 (C・D・E) は単旋律向け。多声の曲では低音などに引きずられる。
- テンポは倍・半分に取り違えることがある。調は平行調 (C メジャーと A マイナー等) を取り違えることがある。
- 正規化範囲・印象の重み・パーツの min/max は暫定値。
