# 使用指南

## 环境与安装

完整重建目前支持 Linux、NVIDIA GPU、CUDA 12 兼容驱动、Python 3.10+。先安装系统工具 `python3-venv git curl`（具体包名随发行版变化），再从仓库根目录执行；Blender、PyBullet 均不是 OBJ 导出的必需依赖：

```bash
./scripts/setup_all.sh
```

脚本创建 `.venv`，安装项目与重建依赖，固定 SAM 2 的 Git 版本，下载并校验 SAM 2.1 tiny 权重。默认不安装碰撞分解、PyBullet 或测试工具，不检查 Blender。安装需要网络和较大的下载空间。PyCOLMAP 的 CPU 和 CUDA 包导入名相同，不要同时安装；默认脚本选择 CUDA 包。`requirements-tested.txt` 包含开发机的可选工具，是版本快照，不是最小安装清单或跨平台锁文件。

可选安装：`./scripts/setup_all.sh --with-collision` 或 `--with-pybullet`。已安装重建环境后可直接 `.venv/bin/pip install -e '.[collision]'` 或 `'.[pybullet]'`，无需重装。Blender 由系统包管理器另行安装，仅生成展示、合成测试素材或独立导入验证时使用。

本项目按 **Git checkout + editable install** 使用，Blender 脚本与参照板位于仓库内；暂不作为独立 PyPI wheel 分发。其他机器不应直接复制 `.venv`。

## 输入与运行

默认参照板与拍摄方式见[拍摄指南](capture.zh-CN.md)。每段视频一件物品，文件名即物品名；也兼容每件物品一个照片子目录。输入目录之外另设输出目录。

```bash
./run-videos.sh input output_me
./run-videos.sh input output_me examples/settings.json
```

按需执行可加 `--objects keyboard` 或 `--objects keyboard.mp4 bottle.mp4`；不指定时执行全部。优先匹配完整文件名，其次匹配唯一的去扩展名名称或物体 ID。同名 `.mp4`／`.mov` 必须带扩展名，未知名称或歧义在执行前报错。带空格的名称用引号括起。

选择执行会共享原输出根目录的 `reconstruction/` 和 `meshy/` 缓存，但将本次模型、进度、日志、场景包和报告写入 `selected/<选择ID>/`，不覆盖原全量交付。相同选择重复运行可复用缓存；终端和 `run-report.json` 的 `output_dir` 给出实际路径。未选择的物品不会提交重建或 Meshy 任务。全量 settings 可继续使用，无需删掉未选物体的设置。

第二条命令的 settings 中只保留本次存在的文件名。`open_container` 用于杯碗：若未观测到开口，程序拒绝导出实心替代物。`outer_shape` 用于普通物体。默认 OBJ-only，不产生碰撞体。

```bash
./run-videos.sh input output_me --with-blender  # 可选展示
./run-videos.sh input output_me --with-collision  # 只增加碰撞块/URDF，无仿真
./run-videos.sh input output_me --pybullet  # 碰撞/URDF + 仿真检查
./run-a4-fast.sh input output_me       # 保留的显式 PyBullet 入口
./run-a4.sh input output_standard     # 较慢的 standard 重建
```

PyBullet 输出中，未填写质量时用 0.1 kg 预览值；真实动力学还需质量、重心、摩擦等信息。快慢档、输入照片或板定义发生变化时，用新的输出目录。

`run` 默认只输出 OBJ 和平滑版；`--obj-only`、`--no-blender` 保留为兼容选项。Blender 展示与碰撞／仿真可独立组合。`assemble` 默认生成 URDF、不运行仿真，需 `--pybullet` 才检查物理场景；单独 `export-pybullet` 只生成文件，不要求安装 PyBullet 引擎。跳过展示／仿真不会跳过米制定标、照片一致性、简化误差或导出水密检查。

## 进度与日志

`run` 每件依次执行重建 → 可选 Meshy → 导出／检查，完成就将普通 OBJ、平滑 OBJ 和 `status.json` 写入 `models/物体名/`，然后才开始下一件。`models/index.json` 同步记录待处理 `pending`、已导出 `needs_review` 和失败 `failed`。`needs_review` 表示模型已交付但几何仍待查看，不是只完成重建。

`run-report.json` 运行中持续记录 `exported`、`failed`、`ready_objects`。后续失败或中断不会撤回之前已交付的物体；Ctrl+C 时写入 `interrupted` 状态。共用场景／ZIP 以及可选整场仿真仍在最后生成。单独的 `reconstruct`、`scan-folder` 命令只负责重建，不承担最终模型批量导出。

终端默认显示动态进度；输出被重定向到文件时自动变成普通日志行。当前物品、当前阶段、缓存标识、阶段与总耗时均可见。抽帧、SAM 分割、轮廓融合和深度剔除显示实际计数；原生深度重建没有可用的百分比回调，显示运行阶段和不断更新的计时。

主流程进度条表示**已完成导出或已明确失败的物品数**，成功与失败在报告中分别统计。当前物体仍在导出时不会提前计入完成数量。场景阶段按渲染和打包任务计数。程序不会用估计时间伪装真实百分比。

```bash
# 包装脚本也支持通过环境变量选择显示方式
PACKING_ASSETS_PROGRESS=plain ./run-videos.sh input output_me
PACKING_ASSETS_PROGRESS=off ./run-videos.sh input output_me

# 更完整的 CLI 控制
.venv/bin/packing-assets run --input input --out output_me \
  --board calibration/a4-dense/board.json --quality fast --obj-only \
  --progress plain --no-blender --json
```

- `progress.json`：原子写入的当前状态，每秒更新；包含当前阶段、物品计数、阶段内部实际计数、缓存标志和耗时。
- `run.log`：默认追加保存本次 Python 和原生算法输出，每次运行有独立时间标记。
- `run-report.json`：最终状态、错误、耗时、缓存复用信息。
- `--verbose`：直接在终端显示算法日志；适合调试，此时不隐藏输出。
- `--json`：最终报告以 JSON 输出到 stdout；进度写入 stderr。

常用入口 `run` 的退出码：完整导出为 0，部分失败或全部失败为 1。`complete_needs_review` 仍表示几何需要查看，不是精度认证。

## 选择正确的输出

最终文件在 `models/视频名/`，普通版 `.obj`、平滑版 `_smooth.obj`，全部米制、Z 向上。`status.json` 记录最终文件重新读取后的边界边、非流形边、封闭体积和 SHA256。原始高面数测量表面位于 `reconstruction/`，可能有孔和缺失底面，不能当最终交付。

默认生成无需 Blender 的 `scene.json` 布局数据。`--with-blender` 才生成 `scene.blend`、`scene-smooth.blend` 和展示图，保持同尺度摆放，只有平移，没有逐件缩放。`scene-package.zip` 携带最终模型、实际生成的场景与报告，不包含重建缓存。

## 恢复与故障

同一命令可以重跑：完成的阶段复用缓存；更新分割或体积算法时会自动隔离受影响的旧缓存。断电时未完成的深度图会先移到 recovery，再重新计算；不要手工把未完成阶段标成 done。旧输入与模型请自行备份，勿同时写同一个输出目录。

已启动的旧版本进程不会自动热更新执行顺序。下一次运行相同命令会使用逐件交付流程并复用缓存。重跑时旧交付会先标为本次待处理，重新检查后再发布，避免把上一轮的结果冒充本次成功。

失败先看 `models/物品名/status.json`、`run-report.json` 和 `run.log`。配准失败通常需增加清晰重叠视角；定标失败检查打印比例、纸张平整度、定位码是否分散可见；模型被质量门槛拒绝需检查分割、薄结构和遮挡。若物体只有很少视角可见，程序不会凭空恢复。

## 可选 Meshy 工具

默认 `--backend local` 不会调用商业 API。`run --backend meshy` 或逐物体 settings 中的 `backend: meshy` 启用付费分支，共用抽帧、相机和 A4 定标结果，接入相同的进度与交付格式。对已有重建可单独运行 `packing-assets meshy --result ... --out ...`。旧脚本仍可用。

详细命令、逐物体切换、费用、真值基准和快速前端见 [Meshy 分支说明](meshy.md)。Meshy 默认跳过稠密重建，使用 `reconstruction-meshy-fast/` 和 `meshy-fast/` 缓存；不生成平滑版或第二套 Blender 场景。可用 `--with-smooth` 开启平滑，`--meshy-frontend full` 恢复完整前端。local 的平滑输出不变。杯碗内腔继续使用 `local`。切换分支会替换 `models/` 交付，但保留各路线缓存；需保留两套交付时先备份旧场景 ZIP。
