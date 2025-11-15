from .open_clip_wrapper import (
    init_open_clip,
    freeze_open_clip_backbone,
    TextEncoderOpenCLIP,
    DiseaseClipDetector,
    ClipClassifierDetector
)

__all__ = [
    "init_open_clip",
    "freeze_open_clip_backbone",
    "TextEncoderOpenCLIP",
    "DiseaseClipDetector",
    "ClipClassifierDetector",
]
