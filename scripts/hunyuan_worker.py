"""Isolated CUDA worker for the optional Hunyuan3D-2mv Turbo experiment."""

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--variant", choices=("turbo", "standard"), default="turbo")
    parser.add_argument("--vae-precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dgen.shapegen.models.autoencoders import ShapeVAE
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan requires a working CUDA PyTorch environment")
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=True)
    steps = 5 if args.variant == "turbo" else 50
    report = {
        "backend": "hunyuan3d-2mv-" + args.variant,
        "status": "loading",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "seed": args.seed,
        "steps": steps,
        "resolution": args.resolution,
        "vae_precision": args.vae_precision,
        "units": "unknown_generated_scale",
        "up_axis": "y",
    }

    def save():
        (args.out / "generation.json").write_text(json.dumps(report, indent=2))

    save()
    start = time.perf_counter()
    try:
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(args.weights.resolve()),
            subfolder="hunyuan3d-dit-v2-mv"
            + ("-turbo" if args.variant == "turbo" else ""),
            variant="fp16",
            use_safetensors=True,
        )
        if args.variant == "turbo":
            pipe.vae = ShapeVAE.from_pretrained(
                str(args.weights.resolve()),
                subfolder="hunyuan3d-vae-v2-0-turbo",
                variant="fp16",
                use_safetensors=True,
                device="cuda",
            )
        if args.variant == "turbo":
            pipe.enable_flashvdm(replace_vae=False, mc_algo="mc")
        # Standard uses the official dense decoder. The optional hierarchical
        # decoder in the pinned upstream casts voxel coordinates to integers.
        torch.cuda.synchronize()
        report["load_seconds"] = time.perf_counter() - start
        prepared = json.loads(args.inputs.read_text())
        images = {
            row["view_tag"]: Image.open(args.inputs.parent / row["file"]).convert(
                "RGBA"
            )
            for row in prepared["views"]
        }
        report["status"] = "generating"
        save()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        latents = pipe(
            image=images,
            num_inference_steps=steps,
            octree_resolution=args.resolution,
            num_chunks=8000,
            generator=torch.manual_seed(args.seed),
            output_type="latent",
        )
        # Optional precision comparison; no hole filling or convex hull.
        decode_dtype = torch.float32 if args.vae_precision == "fp32" else torch.float16
        pipe.vae.to(dtype=decode_dtype)
        with torch.inference_mode():
            mesh = pipe._export(
                latents.to(dtype=decode_dtype),
                output_type="trimesh",
                octree_resolution=args.resolution,
                num_chunks=8000,
            )[0]
        torch.cuda.synchronize()
        report["generation_seconds"] = time.perf_counter() - started
        if mesh is None or not len(mesh.faces):
            raise RuntimeError("model returned no mesh")
        mesh.export(args.out / "source.glb")
        mesh.export(args.out / "raw.obj")
        report.update(
            status="generated_unscaled",
            faces=len(mesh.faces),
            watertight=bool(mesh.is_watertight),
            closed_volume=bool(mesh.is_volume),
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 1024**3,
        )
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - start
        save()


if __name__ == "__main__":
    main()
