# pyright: basic, reportMissingImports=false, reportMissingModuleSource=false
"""Docling Heron layout detection on rendered page images (optional `layout` extra: torch, transformers).

Runs the RT-DETRv2 model directly through transformers rather than the Docling converter, so every region is
kept on scans too (Docling's converter drops text regions without OCR). The model resizes each page to 640x640
itself, so the rendering resolution does not change its input much.
"""

from collections.abc import Sequence
from typing import Any

from PIL import Image

from document_prep.models import Box, LayoutRegion

HERON_REPO = "docling-project/docling-layout-heron"
HERON_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"  # the revision table_benchmark's detections used


class HeronLayoutDetector:
    """Layout regions of page images. Loads the pinned model once; `detect` takes a batch of pages."""

    def __init__(self, *, device: str = "cuda", threshold: float = 0.3, batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModelForObjectDetection, RTDetrImageProcessorPil

        self.model_id = f"{HERON_REPO}@{HERON_REVISION}"
        self.threshold = threshold
        self.batch_size = batch_size
        self._torch = torch
        self._device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self._processor: Any = RTDetrImageProcessorPil.from_pretrained(HERON_REPO, revision=HERON_REVISION)
        model: Any = AutoModelForObjectDetection.from_pretrained(HERON_REPO, revision=HERON_REVISION)
        self._model: Any = model.to(self._device).eval()
        self._labels: dict[int, str] = {int(k): v for k, v in model.config.id2label.items()}

    def detect(self, images: Sequence[Image.Image]) -> list[list[LayoutRegion]]:
        """Regions with score >= threshold for each image, boxes in that image's pixels."""
        results: list[list[LayoutRegion]] = []
        for start in range(0, len(images), self.batch_size):
            batch = [image.convert("RGB") for image in images[start : start + self.batch_size]]
            inputs = self._processor(images=batch, return_tensors="pt").to(self._device)
            with self._torch.inference_mode():
                outputs = self._model(**inputs)
            sizes = [(image.height, image.width) for image in batch]
            detections = self._processor.post_process_object_detection(
                outputs, threshold=self.threshold, target_sizes=sizes
            )
            for found in detections:
                regions = [
                    LayoutRegion(
                        label=self._labels[int(label)],
                        score=round(float(score), 4),
                        box=Box(left=float(b[0]), top=float(b[1]), right=float(b[2]), bottom=float(b[3])),
                    )
                    for score, label, b in zip(found["scores"], found["labels"], found["boxes"].tolist(), strict=True)
                ]
                results.append(regions)
        return results
