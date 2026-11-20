"""層間で共有するデータ構造。ロジックは持たない。"""

from musicchar.models.audio import AudioData
from musicchar.models.character import CharacterParams, PartParam
from musicchar.models.config import (
    AppConfig,
    ArticulationConfig,
    AudioConfig,
    ChordConfig,
    ChromaConfig,
    FeaturesConfig,
    ImpressionInput,
    ImpressionRule,
    KeyConfig,
    MelodyConfig,
    NormRule,
    PartRule,
    PartSource,
)
from musicchar.models.errors import (
    AudioLoadError,
    AudioTooShortError,
    ConfigError,
    FeatureExtractionError,
    ImpressionError,
    MusicCharError,
    RenderError,
)
from musicchar.models.features import (
    Articulation,
    Chords,
    Dynamics,
    FeatureSet,
    KeyMode,
    Melody,
    PitchCenter,
    PitchRange,
    Tempo,
)
from musicchar.models.impressions import IMPRESSION_IDS, ImpressionSet
from musicchar.models.render_config import (
    BodyConfig,
    DecorationSlot,
    DecorationsConfig,
    EffectSlot,
    EffectsConfig,
    BrowsConfig,
    CanvasConfig,
    CheeksConfig,
    ColorsConfig,
    EyesConfig,
    HeadConfig,
    MouthConfig,
    RenderConfig,
)

__all__ = [
    "AudioData", "CharacterParams", "PartParam", "AppConfig", "AudioConfig", "FeaturesConfig", "MelodyConfig", "ChordConfig", "ChromaConfig", "KeyConfig", "ArticulationConfig",
    "ImpressionInput", "ImpressionRule", "NormRule", "PartRule", "PartSource",
    "AudioLoadError", "AudioTooShortError", "ConfigError", "FeatureExtractionError",
    "ImpressionError", "MusicCharError", "RenderError", "Articulation", "Chords",
    "Dynamics", "FeatureSet", "KeyMode", "Melody", "PitchCenter", "PitchRange",
    "Tempo", "IMPRESSION_IDS", "ImpressionSet", "BodyConfig", "DecorationSlot", "DecorationsConfig", "EffectSlot", "EffectsConfig", "BrowsConfig", "CanvasConfig", "CheeksConfig",
    "ColorsConfig", "EyesConfig", "HeadConfig", "MouthConfig", "RenderConfig",
]
