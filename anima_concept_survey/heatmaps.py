from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch

from .progress import ProgressInfo
from .records import public_concept_record


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[AnimaConceptSurvey]"
PREVIEW_NORMALIZATION = "per_file_minmax"


@dataclass
class HeatmapAccumulator:
    token_index: int
    branch: str
    heatmap_sum: torch.Tensor
    count: int = 0
    score_sum: float = 0.0
    score_max: float = 0.0
    token_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptHeatmapAccumulator:
    concept_uid: str
    term: str
    normalized_term: str
    token_source: str
    occurrence_index: int
    branch: str
    token_indices: tuple[int, ...]
    source_token_indices: tuple[int, ...]
    token_texts: tuple[str, ...]
    token_sources: tuple[str, ...]
    token_ids: tuple[int | None, ...]
    heatmap_sum: torch.Tensor
    count: int = 0
    score_sum: float = 0.0
    score_max: float = 0.0


class HeatmapStore:
    def __init__(self, heatmap_dir: str | None):
        self.heatmap_dir = heatmap_dir
        self.token_accumulators: dict[tuple[str, int], HeatmapAccumulator] = {}
        self.concept_accumulators: dict[tuple[str, str], ConceptHeatmapAccumulator] = {}

    def save_token_heatmaps(
        self,
        attention_probs: torch.Tensor,
        token_scores: list[dict[str, Any]],
        spatial: tuple[int, int],
        progress: ProgressInfo,
        eligible_call_index: int,
        branch: str,
        block: Any,
    ) -> None:
        import numpy as np

        out_dir = Path(self.heatmap_dir or "")
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for token in token_scores:
            token_index = int(token["token_index"])
            heatmap_tensor = heatmap_for_token(attention_probs, token_index, spatial).detach().cpu().to(torch.float32)
            heatmap = heatmap_tensor.numpy()
            token_label = safe_filename_text(str(token.get("token_text") or ""))
            suffix = f"_{token_label}" if token_label else ""
            stem = f"step{progress.index:03d}_call{eligible_call_index:03d}_{branch}_token{token_index:03d}{suffix}"
            np.save(out_dir / f"{stem}.npy", heatmap)
            save_heatmap_png(out_dir / f"{stem}.png", heatmap)
            save_heatmap_png(out_dir / f"{stem}_preview.png", heatmap, size=(512, 512), color=True)
            self.update_token_accumulator(branch, token, heatmap_tensor)
            manifest_rows.append({
                "png": f"{stem}.png",
                "preview_png": f"{stem}_preview.png",
                "preview_normalization": PREVIEW_NORMALIZATION,
                "npy": f"{stem}.npy",
                "step_index": progress.index,
                "num_steps": progress.num_steps,
                "eligible_call_index": eligible_call_index,
                "branch": branch,
                "block": block,
                "spatial": list(spatial),
                "token": token,
                **heatmap_stats(heatmap),
            })
        write_heatmap_manifest(out_dir, manifest_rows)
        self.save_aggregate_heatmaps()

    def save_concept_heatmaps(
        self,
        concept_scores: list[dict[str, Any]],
        spatial: tuple[int, int],
        progress: ProgressInfo,
        eligible_call_index: int,
        branch: str,
    ) -> None:
        import numpy as np

        if not concept_scores:
            return
        out_dir = Path(self.heatmap_dir or "") / "concepts"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for concept in concept_scores:
            heatmap_tensor = concept["_heatmap"]
            heatmap = heatmap_tensor.numpy()
            concept_label = concept_filename_label(concept)
            stem = f"step{progress.index:03d}_call{eligible_call_index:03d}_{branch}_concept_{concept_label}"
            np.save(out_dir / f"{stem}.npy", heatmap)
            save_heatmap_png(out_dir / f"{stem}.png", heatmap)
            save_heatmap_png(out_dir / f"{stem}_preview.png", heatmap, size=(512, 512), color=True)
            self.update_concept_accumulator(branch, concept, heatmap_tensor)
            manifest_rows.append({
                "png": f"{stem}.png",
                "preview_png": f"{stem}_preview.png",
                "preview_normalization": PREVIEW_NORMALIZATION,
                "npy": f"{stem}.npy",
                "step_index": progress.index,
                "num_steps": progress.num_steps,
                "eligible_call_index": eligible_call_index,
                "branch": branch,
                "spatial": list(spatial),
                **heatmap_stats(heatmap),
                **public_concept_record(concept),
            })
        write_heatmap_manifest(out_dir, manifest_rows)
        self.save_aggregate_concept_heatmaps()

    def update_token_accumulator(self, branch: str, token: dict[str, Any], heatmap: torch.Tensor) -> None:
        key = (branch, int(token["token_index"]))
        acc = self.token_accumulators.get(key)
        if acc is None:
            acc = HeatmapAccumulator(
                token_index=int(token["token_index"]),
                branch=branch,
                heatmap_sum=torch.zeros_like(heatmap, dtype=torch.float32),
                token_meta={
                    key: token.get(key)
                    for key in ("token_id", "token_text", "token_source", "token_weight")
                    if token.get(key) is not None
                },
            )
            self.token_accumulators[key] = acc
        acc.heatmap_sum += heatmap.to(torch.float32)
        acc.count += 1
        score_mean = token.get("score_mean")
        if score_mean is not None:
            acc.score_sum += float(score_mean)
        score_max = token.get("score_max")
        if score_max is not None:
            acc.score_max = max(acc.score_max, float(score_max))

    def update_concept_accumulator(self, branch: str, concept: dict[str, Any], heatmap: torch.Tensor) -> None:
        key = (branch, str(concept["concept_uid"]))
        acc = self.concept_accumulators.get(key)
        if acc is None:
            acc = ConceptHeatmapAccumulator(
                concept_uid=str(concept["concept_uid"]),
                term=str(concept["term"]),
                normalized_term=str(concept.get("normalized_term") or ""),
                token_source=str(concept.get("token_source") or _first_value(concept.get("token_sources")) or ""),
                occurrence_index=int(concept.get("occurrence_index") or 0),
                branch=branch,
                token_indices=tuple(int(index) for index in concept["token_indices"]),
                source_token_indices=tuple(int(index) for index in concept.get("source_token_indices", ())),
                token_texts=tuple(str(text) for text in concept["token_texts"]),
                token_sources=tuple(str(source) for source in concept["token_sources"]),
                token_ids=tuple(_optional_int(token_id) for token_id in concept.get("token_ids", ())),
                heatmap_sum=torch.zeros_like(heatmap, dtype=torch.float32),
            )
            self.concept_accumulators[key] = acc
        acc.heatmap_sum += heatmap.to(torch.float32)
        acc.count += 1
        acc.score_sum += float(concept.get("score_mean") or 0.0)
        acc.score_max = max(acc.score_max, float(concept.get("score_max") or 0.0))

    def save_aggregate_heatmaps(self) -> None:
        import numpy as np

        if not self.token_accumulators:
            return
        out_dir = Path(self.heatmap_dir or "") / "aggregate"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for acc in sorted(self.token_accumulators.values(), key=lambda item: (item.branch, item.token_index)):
            if acc.count <= 0:
                continue
            heatmap = (acc.heatmap_sum / acc.count).numpy()
            token_label = safe_filename_text(str(acc.token_meta.get("token_text") or ""))
            suffix = f"_{token_label}" if token_label else ""
            stem = f"aggregate_{acc.branch}_token{acc.token_index:03d}{suffix}"
            np.save(out_dir / f"{stem}.npy", heatmap)
            save_heatmap_png(out_dir / f"{stem}.png", heatmap)
            save_heatmap_png(out_dir / f"{stem}_preview.png", heatmap, size=(512, 512), color=True)
            manifest.append({
                "png": f"{stem}.png",
                "preview_png": f"{stem}_preview.png",
                "preview_normalization": PREVIEW_NORMALIZATION,
                "npy": f"{stem}.npy",
                "branch": acc.branch,
                "token_index": acc.token_index,
                "observation_count": acc.count,
                "score_mean": acc.score_sum / acc.count if acc.count else None,
                "score_max": acc.score_max,
                **heatmap_stats(heatmap),
                **acc.token_meta,
            })
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def save_aggregate_concept_heatmaps(self) -> None:
        import numpy as np

        if not self.concept_accumulators:
            return
        out_dir = Path(self.heatmap_dir or "") / "concepts" / "aggregate"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for acc in sorted(self.concept_accumulators.values(), key=lambda item: (item.branch, item.concept_uid)):
            if acc.count <= 0:
                continue
            heatmap = (acc.heatmap_sum / acc.count).numpy()
            concept_label = concept_filename_label({
                "concept_uid": acc.concept_uid,
                "term": acc.term,
                "token_source": acc.token_source,
                "occurrence_index": acc.occurrence_index,
                "token_indices": list(acc.token_indices),
            })
            stem = f"aggregate_{acc.branch}_concept_{concept_label}"
            np.save(out_dir / f"{stem}.npy", heatmap)
            save_heatmap_png(out_dir / f"{stem}.png", heatmap)
            save_heatmap_png(out_dir / f"{stem}_preview.png", heatmap, size=(512, 512), color=True)
            manifest.append({
                "png": f"{stem}.png",
                "preview_png": f"{stem}_preview.png",
                "preview_normalization": PREVIEW_NORMALIZATION,
                "npy": f"{stem}.npy",
                "branch": acc.branch,
                "concept_uid": acc.concept_uid,
                "term": acc.term,
                "normalized_term": acc.normalized_term,
                "token_source": acc.token_source,
                "occurrence_index": acc.occurrence_index,
                "token_indices": list(acc.token_indices),
                "source_token_indices": list(acc.source_token_indices),
                "token_texts": list(acc.token_texts),
                "token_sources": list(acc.token_sources),
                "token_ids": list(acc.token_ids),
                "observation_count": acc.count,
                "score_mean": acc.score_sum / acc.count if acc.count else None,
                "score_max": acc.score_max,
                **heatmap_stats(heatmap),
            })
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def heatmap_for_token(attention_probs: torch.Tensor, token_index: int, spatial: tuple[int, int]) -> torch.Tensor:
    heatmap = attention_probs[:, :, :, token_index].mean(dim=(0, 1))
    return heatmap.reshape(spatial)


def safe_filename_text(value: str, max_len: int = 40) -> str:
    if not value:
        return ""
    value = re.sub(r"\s+", "_", value.strip())
    value = re.sub(r"[^A-Za-z0-9_.-]+", "", value)
    return value[:max_len].strip("._-")


def concept_filename_label(concept: dict[str, Any]) -> str:
    source = safe_filename_text(str(concept.get("token_source") or _first_value(concept.get("token_sources")) or "source"))
    term = safe_filename_text(str(concept.get("term") or "concept"))
    occurrence = int(concept.get("occurrence_index") or 0)
    span = token_span_label(concept.get("token_indices") or ())
    label = f"{source}_{term}_occ{occurrence}_tok{span}"
    return safe_filename_text(label, max_len=140) or safe_filename_text(str(concept.get("concept_uid") or "concept"), max_len=140)


def token_span_label(indices: Any) -> str:
    values = [int(index) for index in indices]
    if not values:
        return "none"
    if len(values) == 1:
        return f"{values[0]:03d}"
    return f"{values[0]:03d}-{values[-1]:03d}"


def _first_value(values: Any) -> Any:
    if isinstance(values, (list, tuple)) and values:
        return values[0]
    return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def heatmap_stats(heatmap: Any) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(heatmap, dtype=np.float32)
    mean_value = float(arr.mean())
    max_value = float(arr.max())
    return {
        "heatmap_mean": mean_value,
        "heatmap_max": max_value,
        "heatmap_std": float(arr.std()),
        "heatmap_max_over_mean": max_value / mean_value if mean_value else 0.0,
    }


def write_heatmap_manifest(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = out_dir / "manifest.json"
    existing = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(rows)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")


def save_heatmap_png(path: Path, heatmap: Any, *, size: tuple[int, int] | None = None, color: bool = False) -> None:
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        LOGGER.warning("%s could not import PIL/numpy for PNG heatmap export", LOG_PREFIX)
        return
    arr = np.asarray(heatmap, dtype=np.float32)
    min_value = float(arr.min())
    max_value = float(arr.max())
    if max_value > min_value:
        arr = (arr - min_value) / (max_value - min_value)
    else:
        arr = np.zeros_like(arr)
    if color:
        image = Image.fromarray(colorize_heatmap(arr), mode="RGB")
    else:
        image = Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8), mode="L")
    if size is not None:
        resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
        image = image.resize(size, resample)
    image.save(path)


def colorize_heatmap(arr: Any) -> Any:
    import numpy as np

    x = np.asarray(arr, dtype=np.float32).clip(0.0, 1.0)
    stops = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.12, 0.45],
        [0.0, 0.75, 0.85],
        [1.0, 0.85, 0.0],
        [1.0, 0.12, 0.0],
    ], dtype=np.float32)
    scaled = x * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int32).clip(0, len(stops) - 1)
    upper = np.ceil(scaled).astype(np.int32).clip(0, len(stops) - 1)
    frac = (scaled - lower)[..., None]
    rgb = stops[lower] * (1.0 - frac) + stops[upper] * frac
    return (rgb * 255.0).clip(0, 255).astype(np.uint8)
