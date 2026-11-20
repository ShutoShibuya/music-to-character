"""Music Character Generator の画面 (Streamlit)。

起動: ``streamlit run app.py``

処理は ``musicchar.pipeline``、表示用データの整形は ``musicchar.report`` が担当し、
このファイルは画面の組み立てだけを行う。
"""

from __future__ import annotations

import io
import json

import streamlit as st

from musicchar.config_loader import load_config
from musicchar.models import AppConfig, AudioLoadError, AudioTooShortError, MusicCharError
from musicchar.pipeline import AnalysisResult, analyze
from musicchar.report import feature_rows, impression_rows, part_rows, result_to_dict

ACCEPTED_TYPES = ["wav", "mp3", "flac", "ogg"]


@st.cache_resource
def get_config() -> AppConfig:
    """設定ファイルを一度だけ読み込む。"""
    return load_config()


@st.cache_data(show_spinner="楽曲を分析してキャラクターを生成しています…")
def run_analysis(data: bytes, name: str) -> AnalysisResult:
    """アップロードされた音声を分析する (同じファイルの再実行はキャッシュを使う)。"""
    return analyze(io.BytesIO(data), get_config(), source=name)


def show_impressions(result: AnalysisResult, config: AppConfig) -> None:
    """印象 1〜8 を、左端 (0.0) と右端 (1.0) のラベル付きのバーで表示する。"""
    for row in impression_rows(result.impressions, config.impressions):
        st.markdown(f"**{row['left']}** ← → **{row['right']}**")
        if row["value"] is None:
            st.caption(f"推定不能: {row['note']}")
        else:
            st.progress(row["value"], text=f"{row['value']:.2f}")
            if row["note"]:
                st.caption(row["note"])


def show_features(result: AnalysisResult) -> None:
    """特徴量 A〜H の表と、音量の時間変化。"""
    st.dataframe(feature_rows(result.features), hide_index=True, width="stretch")
    dyn = result.features.f_dynamics
    if dyn is not None and dyn.rms_curve:
        st.caption(f"音量 (RMS) の時間変化 — 1 点 = {dyn.rms_curve_hop_sec * 1000:.0f} ms")
        st.line_chart(list(dyn.rms_curve))


def show_result(result: AnalysisResult, config: AppConfig) -> None:
    """分析結果の全体を表示する。"""
    st.subheader(f"結果: {result.source}")
    st.caption(
        f"読み込み {result.original_duration:.1f} 秒 / 分析対象 {result.analyzed_duration:.1f} 秒 "
        "(前後の無音を除き、長い曲は中央部分のみ)"
    )
    if all(v is None for v in result.impressions.values.values()):
        st.error("すべての印象を計算できませんでした。キャラクターは中立値で描画されています。")
    elif result.features.errors:
        names = ", ".join(result.features.errors)
        st.warning(f"一部の特徴量を抽出できませんでした ({names})。詳細は「特徴量」タブを見てください。")

    left, right = st.columns([2, 3])
    with left:
        st.image(result.svg, width=360)
        st.download_button(
            "キャラクター (SVG) をダウンロード", result.svg, file_name="character.svg", mime="image/svg+xml"
        )
        st.download_button(
            "分析結果 (JSON) をダウンロード",
            json.dumps(result_to_dict(result), ensure_ascii=False, indent=2),
            file_name="analysis.json",
            mime="application/json",
        )
    with right:
        tab_imp, tab_feat, tab_parts = st.tabs(["印象", "特徴量", "キャラクターのパーツ"])
        with tab_imp:
            show_impressions(result, config)
        with tab_feat:
            show_features(result)
        with tab_parts:
            st.dataframe(part_rows(result.character, config.parts), hide_index=True, width="stretch")


def main() -> None:
    """画面全体。"""
    st.set_page_config(page_title="Music Character Generator", layout="wide")
    st.title("Music Character Generator")
    st.caption("楽曲を分析して、その印象に合ったキャラクターを生成します (歌声・歌詞は分析対象外)。")

    uploaded = st.file_uploader("音声ファイルを選択", type=ACCEPTED_TYPES)
    if uploaded is None:
        st.info("WAV / MP3 / FLAC / OGG の音声ファイルをアップロードしてください。")
        return

    try:
        result = run_analysis(uploaded.getvalue(), uploaded.name)
    except AudioLoadError as e:
        st.error(f"音声ファイルを読み込めませんでした: {e}")
        return
    except AudioTooShortError as e:
        st.error(f"分析できない音声です: {e}")
        return
    except MusicCharError as e:
        st.error(f"分析に失敗しました: {e}")
        return
    except Exception as e:  # 想定外の例外でも画面を落とさない
        st.error(f"想定外のエラーが発生しました: {type(e).__name__}: {e}")
        return
    show_result(result, get_config())


if __name__ == "__main__":
    main()
