#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if [ "$#" -lt 1 ] || [ "${1:-}" = "--help" ]; then
  echo "用法: $0 视频文件夹 [输出文件夹] [settings.json] [--backend local|meshy] [其他 run 参数]"
  echo "默认输出到输入目录旁的 <输入目录名>-models，使用 A4 参照板和快速档。"
  echo "meshy 为显式付费分支，可加 --key-file 仓库外密钥文件；仍需本地米制重建。"
  if [ "${1:-}" = "--help" ]; then exit 0; fi
  exit 2
fi
input=$(realpath -- "$1")
if [ ! -d "$input" ]; then
  echo "输入必须是文件夹，每段视频对应一件物品。" >&2
  exit 2
fi
shift
output="${input}-models"
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then output=$1; shift; fi
args=(run --input "$input" --out "$output" --board "$project_root/calibration/a4-dense/board.json" --quality fast)
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then args+=(--settings "$1"); shift; fi
args+=("$@")
echo "输入: $input"
echo "输出: $output"
exec "$project_root/.venv/bin/packing-assets" "${args[@]}"
