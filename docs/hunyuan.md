# Hunyuan3D-2mv 本机试验

状态：已在 RTX 5080 16 GB 上实际生成瓶子、键盘、飞机。保留为实验入口；当前样例尚未达到本项目的尺寸／水密质量要求，未加入手机网页和 `run-videos.sh --backend` 的正式选择。

流程为：已有视频相机与分割缓存 → 四张透明背景视图 → 本机 Hunyuan 生成 → A4 米制定标及照片对齐 → 检查 → 低面数 OBJ。没有调用 Meshy API，也不会读取 Meshy key。首次下载需联网，生成子进程使用离线模式。

## 已测结果

2026-09-24，RTX 5080，PyTorch 2.8.0 + CUDA 12.8，Hunyuan3D-2mv Turbo，5 步、请求分辨率 384（FlashVDM 内部使用 380），固定随机种子 12345，无贴图、无平滑。

| 视频 | 纯生成 | 加载模型 | 缓存输入至 OBJ 总耗时 | 导出面数 | 照片轮廓 IoU 中位数 |
|---|---:|---:|---:|---:|---:|
| 瓶子 | 2.62 秒 | 9.66 秒 | 31.63 秒 | 8,000 | 81.1% |
| 键盘 | 2.39 秒 | 9.67 秒 | 36.17 秒 | 8,000 | 94.3% |
| 飞机 | 2.44 秒 | 10.08 秒 | 28.65 秒 | 8,000 | 77.8% |

三次 PyTorch 峰值分配约 5.51–5.53 GiB，峰值保留约 6.17–6.42 GiB；这不包含桌面等其他进程的显存。

另测标准版 50 步：同一瓶子纯生成 53.11 秒，含模型加载／对齐／导出共 81.80 秒；峰值分配 5.56 GiB。输出约 79 × 96 × 162 mm、7,999 面，轮廓 IoU 中位数 77.9%，仍未通过水密与外形检查。增加步数没有改善这个输入的结果。机器可读记录见 [hunyuan.json](benchmarks/hunyuan.json)。

**这些不是完整视频处理耗时。** 抽帧、A4 相机恢复、分割、深度测量均复用了已有缓存；全新视频仍要付出这些前处理时间。首次环境安装及权重下载也不计入。

瓶子生成约 84 × 84 × 155 mm，同视频本地输出约 68 × 68 × 190 mm，偏矮偏胖；键盘生成约 376 × 129 × 64 mm，本地参考约 379 × 141 × 40 mm，厚度偏大。三者导出均有少量开放边；没有把失败状态改成通过，也没有用凸包代替外形。瓶子 FP32 解码复测约 3.43 秒、峰值分配 6.43 GiB，没有明显改善比例和水密性，因此默认仍为 FP16。

尺寸校正继续使用原有约 10% 的调整范围与照片质量门槛。样例偏差过大时保留原生成结果并报告 `retained_unvalidated`，不会强行拉伸到目标尺寸后冒充精度合格。

本次四视图来自真实环拍视频，按标定方位排序；例如瓶子的拍摄俯角约 43–58°，并非标准正交前／左／后／右视图。视角分布可能影响模型对比例的判断，这属于待验证原因，不是已证实的唯一原因。轮廓指标也无法保证背面厚度正确。没有独立实测尺寸，不能把本地参考称作真值。

## 安装

在项目根目录运行，依赖现有主环境 `.venv`；本安装入口实测 Python 3.12 / Linux / NVIDIA：

```bash
bash scripts/setup_hunyuan.sh
```

独立环境在 `vendor/hunyuan-venv/`，上游代码在 `vendor/Hunyuan3D-2/`，形状权重在 `vendor/hunyuan-models/`。Turbo 权重约 5.34 GB，另需 CUDA/PyTorch 环境磁盘空间。不会安装贴图渲染器，也不要求 Blender 或 PyBullet。

上游代码锁定 `f8db63096c8282cb27354314d896feba5ba6ff8a`；权重下载固定 Hugging Face revision 并核验 SHA-256。代码及权重不进入本项目源码 ZIP，沿用 [上游许可证](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/LICENSE)。

## 复现一件物品

下面路径是占位符，替换为已经完成的视频标定目录和四视图清单：

```bash
.venv/bin/python scripts/run_hunyuan_case.py \
  --result output/reconstruction-meshy-fast/OBJECT_ID \
  --prepared output/meshy-fast/OBJECT_ID/inputs/prepared.json \
  --out outputs/hunyuan-OBJECT_ID
```

`prepared.json` 可以仅通过现有 `prepare_meshy_views` 选帧程序创建，无需提交付费任务。输出目录须是新目录，避免覆盖此前结果。当前入口复用已有标定数据，尚不是从一批原始视频直接执行的第三个正式 backend。

- `inputs/preview.png`：四视图输入，保留前景 alpha。
- `provider/raw.obj`、`source.glb`：生成原模型，**未校准尺度、Y 向上**。
- `models/hunyuan.obj`：定标／照片对齐后的低面数模型，米、Z 向上；必须结合质量报告使用。
- `models/hunyuan_dimensions.obj`：仅在尺寸校正及检查通过时产生。
- `report.json`：耗时、显存、尺寸、轮廓和水密检查；失败模型继续保留。

标准版对照需要另外下载约 4.93 GB：

```bash
vendor/hunyuan-venv/bin/python scripts/fetch_hunyuan_models.py --include-standard
# 在上述运行命令中添加 --variant standard，并使用新的 --out
```

标准版使用 50 步与官方默认密集解码。锁定版本的可选分层解码存在坐标整数转换问题，试验遇到空表面错误，所以没有把该加速路径接入标准版。`--vae-precision fp32` 可额外比较高精度解码，但不代表模型本身精度更高。

本机结果目录：`outputs/hunyuan-bottle/`、`outputs/hunyuan-keyboard/`、`outputs/hunyuan-plane/`；三物体场景在 `outputs/hunyuan-shapes/scene.blend`，瓶子与 Meshy／本地对比在 `outputs/hunyuan-comparison/scene.blend`。这些私有视频结果不打包进源码。
