import argparse
import json
import os
from pathlib import Path

from .board import create_board
from .media import prepare
from .pipeline import run_batch


def add_blender_options(parser):
    options = parser.add_mutually_exclusive_group()
    options.add_argument(
        "--with-blender",
        dest="render",
        action="store_true",
        help="optionally render Blender scenes; requires external Blender",
    )
    options.add_argument(
        "--no-blender",
        dest="render",
        action="store_false",
        help="skip Blender (default; legacy flag)",
    )
    parser.set_defaults(render=False)


def main():
    parser = argparse.ArgumentParser(
        description="Packing asset preparation — experimental, all geometry in meters"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "configure-meshy", help="save only the local credential file path; no API call"
    )
    p.add_argument("--key-file", type=Path, required=True)
    p = sub.add_parser(
        "run",
        help="automatic folder -> RGB reconstruction -> compound URDFs -> Blender/PyBullet scene",
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--board", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--settings", type=Path)
    p.add_argument(
        "--objects",
        nargs="+",
        help="only these filenames/object names; default all inputs",
    )
    p.add_argument(
        "--backend",
        choices=["local", "meshy"],
        default="local",
        help="mesh branch; meshy uses sparse metric frontend and paid API credits",
    )
    p.add_argument(
        "--key-file", type=Path, help="Meshy credential file outside the repository"
    )
    p.add_argument(
        "--meshy-frontend",
        choices=["hybrid", "fast", "full"],
        default="hybrid",
        help="hybrid measures local dimensions with shared caches (default); fast uses sparse/photo fitting; full retains legacy reconstruction",
    )
    p.add_argument(
        "--with-smooth",
        action="store_true",
        help="also create Meshy smoothed variant (default off; local output unchanged)",
    )
    p.add_argument(
        "--quality", choices=["fast", "draft", "standard"], default="standard"
    )
    add_blender_options(p)
    output = p.add_mutually_exclusive_group()
    output.add_argument(
        "--obj-only",
        dest="output_profile",
        action="store_const",
        const="obj",
        help="OBJ/smooth outputs only (default)",
    )
    output.add_argument(
        "--with-collision",
        dest="output_profile",
        action="store_const",
        const="collision",
        help="also export compound collision meshes and URDF; no simulation",
    )
    output.add_argument(
        "--pybullet",
        dest="output_profile",
        action="store_const",
        const="pybullet",
        help="also export collisions/URDF and run a PyBullet smoke test",
    )
    p.set_defaults(output_profile="obj")
    p.add_argument(
        "--progress",
        choices=["auto", "plain", "off"],
        default=os.environ.get("PACKING_ASSETS_PROGRESS", "auto"),
        help="live terminal bar, plain log lines, or no progress display",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="show algorithm logs in the terminal instead of saving them only to run.log",
    )
    p.add_argument(
        "--json", action="store_true", help="print the final JSON report to stdout"
    )
    p = sub.add_parser(
        "meshy",
        help="existing metric reconstruction -> optional paid Meshy -> checked OBJ",
    )
    p.add_argument("--result", type=Path, required=True)
    p.add_argument(
        "--with-smooth",
        action="store_true",
        help="also create smoothed variant (default off)",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--key-file", type=Path)
    add_blender_options(p)
    p.add_argument(
        "--progress",
        choices=["auto", "plain", "off"],
        default=os.environ.get("PACKING_ASSETS_PROGRESS", "auto"),
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser(
        "assemble",
        help="existing reviewed candidates -> compound URDFs and a shared scene",
    )
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    add_blender_options(p)
    p.add_argument(
        "--pybullet",
        action="store_true",
        help="optionally run the PyBullet smoke test after URDF export",
    )
    p = sub.add_parser(
        "board", help="create a printable board (verify printed dimensions once)"
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scale", type=float, default=1.0)
    p = sub.add_parser(
        "prepare", help="video/photos -> selected full frames + marker detections"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, default=100)
    p = sub.add_parser(
        "batch",
        help="run Meshy or metric conversion jobs from JSON; Meshy submission uses credits",
    )
    p.add_argument("manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser(
        "scan-folder", help="batch: one object per child photo folder or video file"
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--board", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--quality", choices=["fast", "draft", "standard"], default="standard"
    )
    p.add_argument(
        "--scope",
        choices=["outer_shape", "open_container", "outer_convex", "full_geometry"],
        default="outer_shape",
    )
    p = sub.add_parser(
        "reconstruct",
        help="RGB video/photos -> metric object mesh using official COLMAP GPU pipeline",
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--board", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--quality", choices=["fast", "draft", "standard"], default="standard"
    )
    p.add_argument(
        "--segmentation",
        choices=["sam2", "sam2-base", "grabcut", "grabcut-core", "hybrid"],
        default="sam2",
    )
    p.add_argument(
        "--scope",
        choices=["outer_shape", "open_container", "outer_convex", "full_geometry"],
        default="outer_shape",
    )
    p.add_argument("--collision", action="store_true")
    p = sub.add_parser(
        "refine-segmentation",
        help="reuse verified geometry caches with another SAM model; source folder must be retained",
    )
    p.add_argument("--result", type=Path, required=True)
    p.add_argument(
        "--scope",
        choices=["outer_shape", "open_container", "outer_convex", "full_geometry"],
    )
    p.add_argument("--board", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--segmentation",
        choices=["sam2", "sam2-base", "grabcut", "grabcut-core", "hybrid"],
        default="sam2-base",
    )
    p = sub.add_parser(
        "collision",
        help="closed shape -> compound convex collision parts in a shared metric frame",
    )
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threshold-mm", type=float, default=3.0)
    p = sub.add_parser(
        "export-pybullet",
        help="portable single-rigid-body URDF with compound convex collisions",
    )
    p.add_argument("--asset", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mass-kg", type=float, required=True)
    p.add_argument("--threshold-mm", type=float, default=3.0)
    p.add_argument(
        "--no-smooth", action="store_true", help="skip optional smoothed geometry"
    )
    p = sub.add_parser(
        "evaluate",
        help="compare exported Z-up OBJ against an independent ground truth mesh",
    )
    p.add_argument("--mesh", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument(
        "--scope", choices=["outer_convex", "full_geometry"], default="full_geometry"
    )
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser(
        "proxy",
        help="make a closed metric silhouette proxy from a completed reconstruction",
    )
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--board", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pitch", type=float, default=0.0015)
    args = parser.parse_args()
    if args.command == "configure-meshy":
        from .meshy import configure_key_file

        config = configure_key_file(args.key_file)
        print(f"Meshy 密钥文件路径已保存：{config}（未复制密钥，未调用 API）")
    elif args.command == "run":
        from .automation import run_automatic

        result = run_automatic(
            args.input,
            args.board,
            args.out,
            args.settings,
            args.quality,
            args.render,
            args.output_profile == "obj",
            args.progress,
            args.verbose,
            backend=args.backend,
            key_file=args.key_file,
            physics_test=args.output_profile == "pybullet",
            objects=args.objects,
            meshy_frontend=args.meshy_frontend,
            meshy_smooth=args.with_smooth,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                f"完成：{result['exported']} 件导出，{result['failed']} 项失败；状态 {result['status']}"
            )
            output_dir = Path(result.get("output_dir", args.out.resolve()))
            print(
                f"模型：{output_dir / 'models'}\n报告：{output_dir / 'run-report.json'}\n详细日志：{output_dir / 'run.log'}"
            )
        if result["status"] != "complete_needs_review":
            raise SystemExit(1)
    elif args.command == "meshy":
        from .meshy_workflow import run_case

        result = run_case(
            args.result,
            args.out,
            args.key_file,
            args.render,
            args.progress,
            args.verbose,
            smooth=args.with_smooth,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(
                f"Meshy：{result['status']}；报告：{args.out.resolve() / 'run-report.json'}"
            )
            print(f"模型：{args.out.resolve() / 'delivery/models'}")
        if result["status"] != "complete_needs_review":
            raise SystemExit(1)
    elif args.command == "assemble":
        from .automation import assemble

        data = json.loads(args.manifest.read_text())
        for item in data["objects"]:
            item["asset"] = str(
                (args.manifest.resolve().parent / item["asset"]).resolve()
            )
        result = assemble(
            data["objects"],
            args.out,
            data.get("threshold_mm", 3.0),
            args.render,
            physics_test=args.pybullet,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "complete_needs_review":
            raise SystemExit(1)
    elif args.command == "board":
        create_board(args.out, args.scale)
        print(f"Created {args.out.resolve()}/board.svg and board.json")
    elif args.command == "prepare":
        result = prepare(args.input, args.out, args.frames)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "frames"},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "scan-folder":
        from .folder_batch import run_folder

        results = run_folder(args.input, args.board, args.out, args.quality, args.scope)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        if any(r["status"] == "failed" for r in results):
            raise SystemExit(1)
        if any(r["status"] == "rejected" for r in results):
            raise SystemExit(2)
    elif args.command == "reconstruct":
        from .reconstruct import reconstruct

        result = reconstruct(
            args.input,
            args.board,
            args.out,
            args.quality,
            args.collision,
            args.segmentation,
            args.scope,
        )
        status = result.get("selection", {}).get("status", result["status"])
        print(
            json.dumps(
                {
                    "status": status,
                    "surface_status": result["status"],
                    "selection": result.get("selection"),
                    "asset": str(args.out.resolve()),
                },
                indent=2,
            )
        )
        if status == "rejected":
            raise SystemExit(2)
    elif args.command == "refine-segmentation":
        from .refine import refine_segmentation

        result = refine_segmentation(
            args.result, args.board, args.out, args.segmentation, args.scope
        )
        print(json.dumps(result["selection"], indent=2))
        if result["selection"]["status"] == "rejected":
            raise SystemExit(2)
    elif args.command == "collision":
        from .collision import build_collision

        print(
            json.dumps(
                build_collision(args.mesh, args.out, args.threshold_mm / 1000), indent=2
            )
        )
    elif args.command == "export-pybullet":
        from .pybullet_asset import export_pybullet

        print(
            json.dumps(
                export_pybullet(
                    args.asset,
                    args.out,
                    args.mass_kg,
                    args.threshold_mm / 1000,
                    smooth=not args.no_smooth,
                ),
                indent=2,
            )
        )
    elif args.command == "evaluate":
        from .evaluate import evaluate

        print(
            json.dumps(evaluate(args.mesh, args.truth, args.out, args.scope), indent=2)
        )
    elif args.command == "proxy":
        from .proxy import build_proxy

        result = build_proxy(
            args.work, json.loads(args.board.read_text()), args.out, args.pitch
        )
        print(
            json.dumps(
                {
                    "dimensions_m": result["extents_m_xyz_z_up"],
                    "watertight": result["watertight"],
                },
                indent=2,
            )
        )
    else:
        results = run_batch(args.manifest, args.out)
        print(json.dumps(results, ensure_ascii=False, indent=2))
        if any(r["status"] == "failed" for r in results):
            raise SystemExit(1)
        if any(r["status"] == "rejected" for r in results):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
