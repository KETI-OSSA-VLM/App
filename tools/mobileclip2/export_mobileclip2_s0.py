import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ROOT / "vendor" / "ml-mobileclip"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

import open_clip  # noqa: E402
from mobileclip.modules.common.mobileone import reparameterize_model  # noqa: E402


MODEL_NAME = "MobileCLIP2-S0"
PRETRAINED_TAG = "dfndr2b"
DEFAULT_IMAGE_SIZE = 256
DEFAULT_OPSET = 17


@dataclass
class ExportManifest:
    model_name: str
    pretrained: str
    input_shape: list[int]
    output_shape: list[int]
    pytorch_output_path: str
    onnx_path: str
    tflite_path: str | None
    tflite_status: str
    tflite_log_path: str | None


class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MobileCLIP2-S0 image encoder")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "mobileclip2_s0"))
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--stop-after-onnx", action="store_true")
    return parser.parse_args()


def load_model() -> torch.nn.Module:
    model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}
    model, _, _ = open_clip.create_model_and_transforms(
        MODEL_NAME,
        pretrained=PRETRAINED_TAG,
        **model_kwargs,
    )
    model.eval()
    model = reparameterize_model(model)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_dir: Path, image_size: int, opset: int) -> tuple[Path, list[int], list[int], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper = ImageEncoderWrapper(model)
    dummy = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    with torch.no_grad():
        pytorch_output = wrapper(dummy)

    pytorch_output_path = output_dir / "pytorch_output.json"
    pytorch_output_path.write_text(
        json.dumps(
            {
                "input_shape": list(dummy.shape),
                "output_shape": list(pytorch_output.shape),
                "sample_values": [float(x) for x in pytorch_output[0, :8]],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    onnx_path = output_dir / "mobileclip2_s0_image_encoder.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes=None,
        opset_version=opset,
        do_constant_folding=True,
    )

    return onnx_path, list(dummy.shape), list(pytorch_output.shape), pytorch_output_path


def verify_onnx(onnx_path: Path, image_size: int) -> None:
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise RuntimeError(f"Unexpected ONNX signature: {len(inputs)} inputs, {len(outputs)} outputs")
    if tuple(inputs[0].shape) != (1, 3, image_size, image_size):
        raise RuntimeError(f"Unexpected ONNX input shape: {inputs[0].shape}")
    if tuple(outputs[0].shape) != (1, 512):
        raise RuntimeError(f"Unexpected ONNX output shape: {outputs[0].shape}")


def convert_to_tflite(onnx_path: Path, output_dir: Path) -> tuple[str, Path | None, Path | None]:
    tflite_dir = output_dir / "onnx2tf"
    tflite_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "tflite_conversion.log"
    command = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(tflite_dir),
        "-osd",
        "-coion",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    log_path.write_text(
        "# STDOUT\n" + result.stdout + "\n# STDERR\n" + result.stderr,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return f"failed ({result.returncode})", None, log_path

    candidates = sorted(tflite_dir.rglob("*.tflite"))
    if not candidates:
        return "failed (no tflite emitted)", None, log_path
    return "success", candidates[0], log_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    model = load_model()
    onnx_path, input_shape, output_shape, pytorch_output_path = export_onnx(
        model=model,
        output_dir=output_dir,
        image_size=args.image_size,
        opset=args.opset,
    )
    verify_onnx(onnx_path, args.image_size)

    tflite_status = "skipped"
    tflite_path = None
    log_path = None
    if not args.stop_after_onnx:
        tflite_status, tflite_path, log_path = convert_to_tflite(onnx_path, output_dir)

    manifest = ExportManifest(
        model_name=MODEL_NAME,
        pretrained=PRETRAINED_TAG,
        input_shape=input_shape,
        output_shape=output_shape,
        pytorch_output_path=str(pytorch_output_path),
        onnx_path=str(onnx_path),
        tflite_path=str(tflite_path) if tflite_path else None,
        tflite_status=tflite_status,
        tflite_log_path=str(log_path) if log_path else None,
    )
    (output_dir / "manifest.json").write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    print(json.dumps(asdict(manifest), indent=2))


if __name__ == "__main__":
    main()