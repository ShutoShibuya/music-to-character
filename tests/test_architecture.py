"""層の独立性の検査: audio / features / impressions / character は互いに依存しない。

仕様: 「音声解析・印象計算・キャラクター生成を独立させる」。
各層が import してよい自作モジュールを限定し、統合は pipeline / app だけが行う。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "musicchar"

# 層 -> import を許す自作パッケージの接頭辞
ALLOWED = {
    "models": ("musicchar.models",),
    "audio": ("musicchar.models", "musicchar.audio"),
    "features": ("musicchar.models", "musicchar.features"),
    "impressions": ("musicchar.models", "musicchar.impressions"),
    "character": ("musicchar.models", "musicchar.character"),
}


def internal_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
    return {m for m in found if m == "musicchar" or m.startswith("musicchar.")}


@pytest.mark.parametrize("layer", sorted(ALLOWED))
def test_layer_imports_only_what_it_may(layer: str) -> None:
    files = list((PKG / layer).rglob("*.py"))
    assert files, layer
    for path in files:
        for module in internal_imports(path):
            assert module.startswith(ALLOWED[layer]), f"{path.relative_to(ROOT)} imports {module}"


def test_only_the_integration_layer_depends_on_every_layer() -> None:
    """pipeline が全層をつなぎ、app は pipeline / report / config だけを使う。"""
    pipeline_deps = internal_imports(PKG / "pipeline.py")
    assert {"musicchar.audio", "musicchar.features", "musicchar.impressions", "musicchar.character"} <= pipeline_deps
    app_deps = internal_imports(ROOT / "app.py")
    assert not any(m.startswith(("musicchar.audio", "musicchar.features", "musicchar.impressions",
                                 "musicchar.character")) for m in app_deps), app_deps


def test_config_loader_depends_only_on_models() -> None:
    assert all(m.startswith("musicchar.models") for m in internal_imports(PKG / "config_loader.py"))
