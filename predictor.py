"""Checkpoint-backed tracker implementation.

The evaluation script only expects two public entry points from this module:

- :func:`load_model`
- :func:`run_tracker`

This file keeps the pipeline structure intact while adding:

- compatibility shims for the copied model package under the legacy ``lib.*``
  namespace used inside the checkpoint/model code
- resilient checkpoint loading with ``torch.load``
- a lightweight tracker wrapper that performs template/search inference and
  converts predictions back to image coordinates
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import torch

from download import download_checkpoint


ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "model.pth"
TEMPLATE_SIZE = 128
SEARCH_SIZE = 256
TEMPLATE_FACTOR = 2.0
SEARCH_FACTOR = 4.5
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _package(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__path__ = [str(path)]
    return module


def _placeholder_module(name: str, path: Path) -> types.ModuleType:
    module = _package(name, path)

    def __getattr__(attr: str) -> Any:
        if attr.startswith("__"):
            raise AttributeError(attr)

        class Placeholder:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        Placeholder.__name__ = attr
        Placeholder.__qualname__ = attr
        return Placeholder

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    return module


def _load_module(name: str, path: Path) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module {name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_legacy_namespace() -> None:
    """Register compatibility modules required by the copied model code."""

    _package("lib", ROOT_DIR)
    _package("lib.models", ROOT_DIR / "models")
    _package("lib.models.layers", ROOT_DIR / "models" / "layers")
    _package("lib.models.abavitrack", ROOT_DIR / "models" / "abavitrack")
    _package("lib.utils", ROOT_DIR)
    _package("lib.train", ROOT_DIR)
    _package("lib.train.admin", ROOT_DIR)
    for module_name in (
        "lib.train.admin.settings",
        "lib.train.admin.local",
        "lib.train.admin.loading",
        "lib.train.admin.env",
        "lib.train.admin.optim",
        "lib.train.admin.utils",
    ):
        _placeholder_module(module_name, ROOT_DIR)

    stats_mod = _package("lib.train.admin.stats", ROOT_DIR)
    if not hasattr(stats_mod, "AverageMeter"):
        class AverageMeter:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.val = 0
                self.avg = 0
                self.sum = 0
                self.count = 0

        stats_mod.AverageMeter = AverageMeter

    if not hasattr(stats_mod, "StatValue"):
        class StatValue:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.value = 0

        stats_mod.StatValue = StatValue

    if "lib.utils.box_ops" not in sys.modules:
        box_ops = types.ModuleType("lib.utils.box_ops")

        def box_xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
            x0, y0, x1, y1 = box.unbind(-1)
            return torch.stack(
                ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0),
                dim=-1,
            )

        def box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
            cx, cy, w, h = box.unbind(-1)
            return torch.stack(
                (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                dim=-1,
            )

        box_ops.box_xyxy_to_cxcywh = box_xyxy_to_cxcywh
        box_ops.box_cxcywh_to_xyxy = box_cxcywh_to_xyxy
        sys.modules["lib.utils.box_ops"] = box_ops


def _import_model_stack() -> Tuple[Any, Any, Any]:
    """Import the model package under the legacy namespace used by the checkpoint."""

    layers_dir = ROOT_DIR / "models" / "layers"
    abavitrack_dir = ROOT_DIR / "models" / "abavitrack"

    patch_embed = _load_module(
        "lib.models.layers.patch_embed",
        layers_dir / "patch_embed.py",
    )
    layers_pkg = sys.modules["lib.models.layers"]
    layers_pkg.PatchEmbed = patch_embed.PatchEmbed

    for module_name in ["frozen_bn", "attn", "attn_blocks", "rpe", "head"]:
        _load_module(
            f"lib.models.layers.{module_name}",
            layers_dir / f"{module_name}.py",
        )

    utils_mod = _load_module(
        "lib.models.abavitrack.utils",
        abavitrack_dir / "utils.py",
    )

    for module_name in ["base_backbone", "AbaViT", "vit", "vit_ce"]:
        _load_module(
            f"lib.models.abavitrack.{module_name}",
            abavitrack_dir / f"{module_name}.py",
        )

    abavitrack_mod = _load_module(
        "lib.models.abavitrack.abavitrack",
        abavitrack_dir / "abavitrack.py",
    )

    return abavitrack_mod.build_abavitrack, utils_mod.sample_target, utils_mod.clip_box


def _default_cfg() -> SimpleNamespace:
    """Create the minimum configuration required by ``build_abavitrack``."""

    return SimpleNamespace(
        MODEL=SimpleNamespace(
            PRETRAIN_FILE="",
            BACKBONE=SimpleNamespace(
                TYPE="abavit_patch16_224",
                STRIDE=16,
                CAT_MODE="direct",
                SEP_SEG=False,
            ),
            HEAD=SimpleNamespace(
                TYPE="CENTER",
                NUM_CHANNELS=256,
            ),
            RETURN_INTER=False,
            RETURN_STAGES=[2, 5, 8, 11],
            HIDDEN_DIM=192,
            NUM_CHANNELS=256,
        ),
        DATA=SimpleNamespace(
            SEARCH=SimpleNamespace(SIZE=SEARCH_SIZE),
            TEMPLATE=SimpleNamespace(SIZE=TEMPLATE_SIZE),
        ),
        TRAIN=SimpleNamespace(DROP_PATH_RATE=0.1),
    )


def _extract_state_dict(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()

    if not isinstance(checkpoint, dict):
        return None

    preferred_keys = (
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "network",
        "weights",
        "ema",
        "model_ema",
    )

    for key in preferred_keys:
        value = checkpoint.get(key)
        if isinstance(value, torch.nn.Module):
            return value.state_dict()
        if isinstance(value, dict):
            nested = _extract_state_dict(value)
            if nested is not None:
                return nested

    if checkpoint and all(isinstance(key, str) for key in checkpoint.keys()):
        tensor_values = [torch.is_tensor(value) for value in checkpoint.values()]
        if tensor_values and all(tensor_values):
            return checkpoint  # type: ignore[return-value]

    return None


def _strip_common_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    keys = list(state_dict.keys())
    for prefix in prefixes:
        if keys and all(key.startswith(prefix) for key in keys):
            return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _move_plain_tensor_attrs(module: torch.nn.Module, device: torch.device) -> None:
    """Move tensor attributes that are not registered as parameters or buffers."""

    for name, value in list(vars(module).items()):
        if torch.is_tensor(value):
            setattr(module, name, value.to(device))

    for child in module.children():
        _move_plain_tensor_attrs(child, device)


def _load_checkpoint_object(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class TrackerModel(torch.nn.Module):
    """Thin inference wrapper around the loaded tracking network."""

    def __init__(
        self,
        net: torch.nn.Module,
        device: torch.device,
        sample_target_fn: Any,
        clip_box_fn: Any,
    ) -> None:
        super().__init__()
        self.net = net
        self.device = device
        self.sample_target = sample_target_fn
        self.clip_box = clip_box_fn
        self.template_factor = TEMPLATE_FACTOR
        self.search_factor = SEARCH_FACTOR
        self.template_size = TEMPLATE_SIZE
        self.search_size = SEARCH_SIZE
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.template: Optional[torch.Tensor] = None
        self.prev_box: Optional[list[float]] = None
        self.use_amp = device.type == "cuda"

    def _preprocess(self, crop: Any) -> torch.Tensor:
        tensor = torch.from_numpy(crop).to(self.device, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return (tensor / 255.0 - self.mean) / self.std

    @staticmethod
    def _box_to_crop_meta(box: Iterable[float], factor: float) -> Tuple[float, float, float]:
        x, y, w, h = [float(v) for v in box]
        crop_sz = max(1.0, float(math.ceil((w * h) ** 0.5 * factor)))
        x1 = round(x + 0.5 * w - 0.5 * crop_sz)
        y1 = round(y + 0.5 * h - 0.5 * crop_sz)
        return float(x1), float(y1), crop_sz

    def initialize(self, frame: Any, init_box: Iterable[float]) -> list[float]:
        init_box_list = [float(v) for v in init_box]
        self.prev_box = init_box_list
        self.net.eval()

        crop, _, _ = self.sample_target(
            frame,
            init_box_list,
            self.template_factor,
            output_sz=self.template_size,
        )
        self.template = self._preprocess(crop)
        return init_box_list

    def _forward_net(self, search: torch.Tensor) -> Optional[torch.Tensor]:
        if self.template is None:
            return None

        try:
            output = self.net(self.template, search)
        except TypeError:
            output = self.net(template=self.template, search=search)

        if isinstance(output, dict):
            for key in ("pred_boxes", "bbox", "boxes", "pred_box"):
                value = output.get(key)
                if torch.is_tensor(value):
                    return value
            return None

        if isinstance(output, (tuple, list)) and output:
            for item in output:
                if torch.is_tensor(item):
                    return item
            return None

        if torch.is_tensor(output):
            return output

        return None

    def _decode_prediction(
        self,
        pred: torch.Tensor,
        frame_shape: Tuple[int, int],
        crop_meta: Tuple[float, float, float],
    ) -> list[float]:
        pred = pred.detach()
        if pred.ndim > 1:
            pred = pred.reshape(-1, pred.shape[-1])[0]
        pred = pred.float().cpu()

        if pred.numel() != 4 or not torch.isfinite(pred).all():
            return self.prev_box[:] if self.prev_box is not None else [0.0, 0.0, 0.0, 0.0]

        cx, cy, w, h = pred.tolist()
        x1, y1, crop_sz = crop_meta

        abs_cx = x1 + cx * crop_sz
        abs_cy = y1 + cy * crop_sz
        abs_w = max(1.0, w * crop_sz)
        abs_h = max(1.0, h * crop_sz)

        box = [abs_cx - abs_w / 2, abs_cy - abs_h / 2, abs_w, abs_h]
        return self.clip_box(box, frame_shape[0], frame_shape[1])

    def track(self, frame: Any) -> list[float]:
        if self.prev_box is None:
            raise RuntimeError("TrackerModel.track() called before initialize().")

        search_crop, _, _ = self.sample_target(
            frame,
            self.prev_box,
            self.search_factor,
            output_sz=self.search_size,
        )
        search = self._preprocess(search_crop)
        crop_meta = self._box_to_crop_meta(self.prev_box, self.search_factor)

        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = self._forward_net(search)
            else:
                pred = self._forward_net(search)

        if pred is None:
            return self.prev_box[:]

        next_box = self._decode_prediction(pred, frame.shape[:2], crop_meta)
        self.prev_box = next_box
        return next_box


def _resolve_checkpoint_path() -> Path:
    if CHECKPOINT_PATH.exists():
        return CHECKPOINT_PATH
    download_checkpoint(file_id="YOUR_FILE_ID")
    return CHECKPOINT_PATH


def load_model(device: str = "cuda") -> TrackerModel:
    """Load the tracker and checkpoint once before evaluation."""

    _ensure_legacy_namespace()
    build_abavitrack, sample_target_fn, clip_box_fn = _import_model_stack()

    torch_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
    if torch_device.type == "cuda":
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
    checkpoint_path = _resolve_checkpoint_path()
    try:
        checkpoint = _load_checkpoint_object(checkpoint_path)
    except Exception as exc:
        print(f"Warning: checkpoint load failed ({exc}); using the instantiated model without weight restoration.")
        checkpoint = None

    if isinstance(checkpoint, torch.nn.Module):
        net = checkpoint
    else:
        state_dict = _extract_state_dict(checkpoint)
        cfg = _default_cfg()
        net = build_abavitrack(cfg, training=False)

        if state_dict is not None:
            state_dict = _strip_common_prefixes(state_dict)
            missing_keys, unexpected_keys = net.load_state_dict(state_dict, strict=False)
            if missing_keys or unexpected_keys:
                print(
                    f"Checkpoint loaded with missing={len(missing_keys)} "
                    f"unexpected={len(unexpected_keys)} keys.",
                )
        else:
            print("Checkpoint did not contain a plain state dict; using the instantiated model as-is.")

    net = net.to(torch_device)
    _move_plain_tensor_attrs(net, torch_device)
    net.eval()

    tracker = TrackerModel(
        net=net,
        device=torch_device,
        sample_target_fn=sample_target_fn,
        clip_box_fn=clip_box_fn,
    )
    tracker.eval()
    return tracker


def read_init_box(path: str) -> list[float]:
    """Read the first-frame annotation in ``x,y,w,h`` format."""

    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    x, y, w, h = map(float, line.split(","))
    return [x, y, w, h]


def run_tracker(
    model: Any,
    video_path: str,
    init_box_path: str,
) -> list[dict[str, float]]:
    """Run tracking on one video sequence and return frame-wise predictions."""

    predictions: list[dict[str, float]] = []
    init_box = read_init_box(init_box_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to open video: {video_path}")

    if not hasattr(model, "initialize") or not hasattr(model, "track"):
        raise TypeError("Loaded model must expose initialize() and track() methods.")

    frame_idx = 0
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_idx == 0:
                pred_box = model.initialize(frame, init_box)
            else:
                pred_box = model.track(frame)

            x, y, w, h = [float(v) for v in pred_box]
            predictions.append(
                {
                    "frame_idx": frame_idx,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )
            frame_idx += 1
    finally:
        cap.release()

    return predictions