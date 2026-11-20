"""パーツ値 (CharacterParams) と描画設定から、顔の各要素の座標・大きさ・角度・色を計算する。

SVG 文字列を作る前の数値計算だけを担当する純粋関数で、乱数を使わない。
角度は SVG の ``rotate()`` と同じ向き (正 = 時計回り、y 軸は下向き) で持つ。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from musicchar.character.renderer.color import hsl_to_hex
from musicchar.models.character import CharacterParams
from musicchar.models.errors import RenderError
from musicchar.models.render_config import RenderConfig

# 顔の描画に必要なパーツ名 (character_parts.yaml のキー)
FACE_PARTS = (
    "eye_brow_distance", "cheek_intensity", "eye_angle", "brow_angle", "brow_thickness",
    "brow_shortness", "mouth_corner", "eye_highlight", "eye_height", "eye_spacing",
    "outline_roundness", "mouth_openness", "body_lightness", "body_saturation",
)


# エフェクト・装飾の描画に必要なパーツ名
EXTRA_PARTS = ("effect_count", "decoration_amount")

# 体の描画に必要なパーツ名 (体の色は FACE_PARTS 側の body_lightness / body_saturation)
BODY_PARTS = ("body_width", "body_height", "body_texture")


@dataclass(frozen=True)
class HeadGeom:
    """頭(角丸の矩形)。x, y は左上、corner は角の丸み半径。"""

    x: float
    y: float
    width: float
    height: float
    corner: float


@dataclass(frozen=True)
class EyeGeom:
    """目(楕円)。rot は (cx, cy) まわりの回転角(度)。highlight_* は目の中のハイライト。"""

    cx: float
    cy: float
    rx: float
    ry: float
    rot: float
    highlight_cx: float
    highlight_cy: float
    highlight_r: float


@dataclass(frozen=True)
class BrowGeom:
    """眉(太い線分)。(cx, cy) が中心で、rot はそのまわりの回転角(度)。"""

    cx: float
    cy: float
    length: float
    thickness: float
    rot: float


@dataclass(frozen=True)
class MouthGeom:
    """口。curve > 0 で口角が上がり、open_h は中央での口の開き(縦)。"""

    cx: float
    cy: float
    half_width: float
    curve: float
    open_h: float


@dataclass(frozen=True)
class CheekGeom:
    """ほっぺ(楕円)。"""

    cx: float
    cy: float
    rx: float
    ry: float
    opacity: float


@dataclass(frozen=True)
class FaceLayout:
    """顔の全要素の計算結果。left / right は見ている人から見た左 / 右。"""

    head: HeadGeom
    skin_fill: str
    skin_stroke: str
    eye_left: EyeGeom
    eye_right: EyeGeom
    brow_left: BrowGeom
    brow_right: BrowGeom
    mouth: MouthGeom
    cheek_left: CheekGeom
    cheek_right: CheekGeom


def skin_colors(params: CharacterParams, cfg: RenderConfig) -> tuple[str, str]:
    """頭の塗りと輪郭線の色を返す。体の色 (明度・彩度) と共通で、輪郭線は明度を下げた色。"""
    hue = cfg.colors.hue_deg
    sat = params.value("body_saturation")
    light = params.value("body_lightness")
    fill = hsl_to_hex(hue, sat, light)
    stroke = hsl_to_hex(hue, sat, light - cfg.head.stroke_lightness_delta)
    return fill, stroke


def compute_face_layout(params: CharacterParams, cfg: RenderConfig) -> FaceLayout:
    """顔の各要素の座標・大きさ・角度・色を計算する。

    Args:
        params: 全パーツの値。
        cfg: 描画レイアウト設定。

    Returns:
        ``FaceLayout``。

    Raises:
        RenderError: 描画に必要なパーツ (``FACE_PARTS``) が ``params`` にない場合。
    """
    missing = [n for n in FACE_PARTS if n not in params.parts]
    if missing:
        raise RenderError(f"描画に必要なパーツがありません: {missing}")
    v = params.value
    h, e, b, m, c = cfg.head, cfg.eyes, cfg.brows, cfg.mouth, cfg.cheeks

    max_corner = min(h.width, h.height) / 2
    corner = min(max_corner, h.min_corner) + v("outline_roundness") * (max_corner - min(max_corner, h.min_corner))
    head = HeadGeom(h.cx - h.width / 2, h.cy - h.height / 2, h.width, h.height, corner)
    fill, stroke = skin_colors(params, cfg)

    eye_cy = h.cy + e.y_offset
    half_spacing = e.spacing * v("eye_spacing") / 2
    eye_ry = e.ry * v("eye_height")
    hl_r = e.highlight_max_r * v("eye_highlight")
    angle = v("eye_angle")  # 正 = 外側が上がる(つり目)

    def eye(cx: float, rot: float) -> EyeGeom:
        return EyeGeom(
            cx, eye_cy, e.rx, eye_ry, rot,
            cx + e.rx * e.highlight_dx, eye_cy - eye_ry * e.highlight_dy, hl_r,
        )

    # 左目は外側が左端なので、時計回り(正の rotate)で左端が上がる。右目は逆回り。
    eye_left = eye(h.cx - half_spacing, angle)
    eye_right = eye(h.cx + half_spacing, -angle)

    brow_cy = eye_cy - eye_ry - b.gap * v("eye_brow_distance")
    brow_len = b.length * (1.0 - b.max_shortening * v("brow_shortness"))
    brow_th = b.thickness * v("brow_thickness")
    brow_angle = v("brow_angle")  # 正 = 内側が下がる(怒り眉)
    # 左眉は内側が右端なので、時計回り(正の rotate)で右端が下がる。右眉は逆回り。
    brow_left = BrowGeom(eye_left.cx, brow_cy, brow_len, brow_th, brow_angle)
    brow_right = BrowGeom(eye_right.cx, brow_cy, brow_len, brow_th, -brow_angle)

    mouth = MouthGeom(
        h.cx, h.cy + m.y_offset, m.half_width, v("mouth_corner") * m.curve_depth,
        v("mouth_openness") * m.max_open,
    )

    opacity = c.max_opacity * v("cheek_intensity")
    cheek_cy = h.cy + c.y_offset
    return FaceLayout(
        head=head, skin_fill=fill, skin_stroke=stroke,
        eye_left=eye_left, eye_right=eye_right, brow_left=brow_left, brow_right=brow_right,
        mouth=mouth,
        cheek_left=CheekGeom(h.cx - c.x_offset, cheek_cy, c.rx, c.ry, opacity),
        cheek_right=CheekGeom(h.cx + c.x_offset, cheek_cy, c.rx, c.ry, opacity),
    )


@dataclass(frozen=True)
class BodyGeom:
    """体(角丸の矩形)。x, y は左上、corner は角の丸み半径。"""

    x: float
    y: float
    width: float
    height: float
    corner: float


@dataclass(frozen=True)
class TextureGeom:
    """体のざらつき (ノイズの重ね塗り)。opacity が 0 なら描かない。

    Attributes:
        opacity: ざらつきの濃さ (0〜1)。
        color: 粒の色 (#rrggbb)。体が暗ければ明るく、明るければ暗い色。
        frequency: ノイズの細かさ (feTurbulence の baseFrequency)。大きいほど粒が細かい。
        octaves: ノイズの重ね数。
        seed: ノイズの乱数シード (固定)。
    """

    opacity: float
    color: str
    frequency: float
    octaves: int
    seed: int


@dataclass(frozen=True)
class BodyLayout:
    """体の計算結果。"""

    body: BodyGeom
    fill: str
    stroke: str
    stroke_width: float
    texture: TextureGeom


def compute_body_layout(params: CharacterParams, cfg: RenderConfig) -> BodyLayout:
    """体の座標・大きさ・色・ざらつきを計算する。

    体は頭の中心の真下に置き、上端が頭の下端から ``body.top_overlap`` だけ頭側に重なる
    (体は頭の後ろに描かれる)。幅・高さは基準値に ``body_width`` / ``body_height`` の倍率を掛ける。
    色は頭と同じ色相・彩度で、明度だけ ``body.lightness_shift`` ずらす。

    Raises:
        RenderError: 描画に必要なパーツが ``params`` にない場合。
    """
    missing = [n for n in (*BODY_PARTS, "body_lightness", "body_saturation") if n not in params.parts]
    if missing:
        raise RenderError(f"描画に必要なパーツがありません: {missing}")
    v = params.value
    b, h = cfg.body, cfg.head

    width, height = b.width * v("body_width"), b.height * v("body_height")
    top = h.cy + h.height / 2 - b.top_overlap
    geom = BodyGeom(h.cx - width / 2, top, width, height, b.corner_ratio * min(width, height))

    hue, sat, light = cfg.colors.hue_deg, v("body_saturation"), v("body_lightness")
    fill = hsl_to_hex(hue, sat, light + b.lightness_shift)
    stroke = hsl_to_hex(hue, sat, light + b.lightness_shift - h.stroke_lightness_delta)

    t = min(1.0, max(0.0, v("body_texture")))
    body_lightness = light + b.lightness_shift
    grain_lightness = body_lightness + (b.texture_contrast if body_lightness < 0.5 else -b.texture_contrast)
    texture = TextureGeom(
        opacity=b.texture_max_opacity * t,
        color=hsl_to_hex(hue, sat, grain_lightness),
        frequency=b.texture_freq_min + t * (b.texture_freq_max - b.texture_freq_min),
        octaves=int(b.texture_octaves),
        seed=int(b.texture_seed),
    )
    return BodyLayout(body=geom, fill=fill, stroke=stroke, stroke_width=b.stroke_width, texture=texture)


@dataclass(frozen=True)
class EffectGeom:
    """エフェクト 1 個。x, y はキャンバス座標、rot は度。"""

    kind: str
    x: float
    y: float
    size: float
    rot: float


@dataclass(frozen=True)
class DecorationGeom:
    """装飾 1 個。x, y はキャンバス座標 (頭・体の大きさに合わせて解決済み)、fill は塗りの色。"""

    kind: str
    x: float
    y: float
    size: float
    rot: float
    fill: str


@dataclass(frozen=True)
class ExtrasLayout:
    """エフェクトと装飾の計算結果。個数は ``effect_count`` / ``decoration_amount`` に対応する。"""

    effects: tuple[EffectGeom, ...]
    effect_color: str
    decorations: tuple[DecorationGeom, ...]
    stroke: str
    stroke_width: float


def _count(value: float, available: int) -> int:
    """個数のパーツ値を、四捨五入して 0〜スロット数に収めた整数にする。"""
    return min(available, max(0, int(math.floor(value + 0.5))))


def compute_extras_layout(
    params: CharacterParams, face: FaceLayout, body: BodyLayout, cfg: RenderConfig
) -> ExtrasLayout:
    """エフェクトと装飾の位置・大きさ・色を計算する。

    設定のスロットリストの先頭から、``effect_count`` / ``decoration_amount`` 個を使う
    (乱数は使わない)。装飾の位置は、``anchor`` が ``head`` なら頭、``body`` なら体の
    矩形に対する相対座標 (fx: 左右, fy: 上下) から解決するので、体の幅・高さが変わっても
    体の上に乗る。

    Raises:
        RenderError: 描画に必要なパーツが ``params`` にない場合。
    """
    missing = [n for n in EXTRA_PARTS if n not in params.parts]
    if missing:
        raise RenderError(f"描画に必要なパーツがありません: {missing}")
    fx_cfg, dec_cfg = cfg.effects, cfg.decorations

    n_effects = _count(params.value("effect_count"), len(fx_cfg.slots))
    effects = tuple(EffectGeom(s.kind, s.x, s.y, s.size, s.rot) for s in fx_cfg.slots[:n_effects])

    boxes = {
        "head": (face.head.x, face.head.y, face.head.width, face.head.height),
        "body": (body.body.x, body.body.y, body.body.width, body.body.height),
    }
    accents = {1: dec_cfg.accent_1, 2: dec_cfg.accent_2}
    n_decos = _count(params.value("decoration_amount"), len(dec_cfg.slots))
    decorations = []
    for s in dec_cfg.slots[:n_decos]:
        x0, y0, w, h = boxes[s.anchor]
        decorations.append(
            DecorationGeom(s.kind, x0 + (s.fx + 0.5) * w, y0 + s.fy * h, s.size, s.rot, accents[s.accent])
        )
    return ExtrasLayout(
        effects=effects, effect_color=fx_cfg.color, decorations=tuple(decorations),
        stroke=dec_cfg.stroke, stroke_width=dec_cfg.stroke_width,
    )
