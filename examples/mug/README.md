# Real mug: phone video → metric OBJ

[Input GIF](../../docs/media/mug-input.gif) · [Rendered result](../../docs/media/mug-result.png) · [Download OBJ](mug.obj) · [Validation](validation.json)

这是本项目已有真实手机任务的公开演示：一只马克杯和一张固定 A4 参照板。输入原片为 1920×1080、约 47 秒；演示截取第 2–34 秒，以 4 倍速压缩为 8 秒无声 GIF。画面为杯子、参照板和地面，不附完整原视频、音频、手机元数据或任务数据库。

输出使用本地重建分支及杯类几何修复后的平滑版本，未经人工替换形状。这个案例不是 Meshy 成绩，也不是未经修复的原始摄影测量网格。把手和杯内腔保留；局部外形仍可能偏离实物。

- 文件单位：米；Z 向上；原点为底部中心。
- 包围盒：116.176 × 85.307 × 105.633 mm（A4 视频定标估计，非实测）。
- 闭合性、面数、内腔探测与文件 SHA-256 见 `validation.json`。
- 内腔探测和水密检查不等于几何准确性认证。
- 无贴图：README 渲染使用统一材质，不是实物表面颜色。

GIF 是压缩展示素材，不能用它复现原片重建质量。复现同类流程时，请自行录制原始视频，打印 [A4 参照板](../../calibration/a4-dense/reference-board.pdf)，保持参照板和物体固定；命令如下：

```bash
./scripts/setup_all.sh
mkdir -p input
# 将自己的原始视频命名为 mug.mp4，放入 input/
./run-videos.sh input output_mug examples/mug/settings.json --objects mug
```

模型位于 `output_mug/selected/<选择ID>/models/`（命令会打印实际路径）。无需 Blender 或 PyBullet；需要 Linux/NVIDIA GPU。拍摄方法见[指南](../../docs/capture.zh-CN.md)。

## Preview regeneration

With FFmpeg installed, this recreates the GIF from the same original capture (not bundled):

```bash
ffmpeg -ss 2 -t 32 -i /path/to/mug.mp4 -an \
  -filter_complex '[0:v]setpts=(PTS-STARTPTS)/4,fps=6,scale=420:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=3' \
  -loop 0 docs/media/mug-input.gif
```

The sample mesh and media are curated project demo assets. No API keys, third-party model weights or full recording archive are distributed with them.
