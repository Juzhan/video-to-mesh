# 可切换的 Meshy 分支

Meshy 已纳入公开源码、批处理主入口、进度显示和源码 ZIP。默认仍为 `local`；只有显式选择 `meshy`（命令行或逐物体 settings）才上传图片并调用付费 API。调用的是官方 REST 服务，不要求安装 MCP 服务器。

## 切换方式

混合模式中，Meshy 几何检查未通过，或尺寸拟合报错（例如与本地尺寸相差超过约 10%）时，如果已得到通过自动检查的本地重建候选，会额外发布 `models/物体名/物体名_local.obj` 和 `local-report.json`。该文件沿用米制尺度和低面数导出，保留原 Meshy 模型；手机页面和 ZIP 均可取得。本地候选仍需检查隐藏面、细部和真实尺寸，主任务不会因此掩盖 Meshy 版本的待核验状态。

已有 fast 任务升级 hybrid 时，除补算本地尺寸外，还会生成独立的 `hybrid-metric-*`／`hybrid-photo-*` 对齐候选，并使用本地密集重建重新计算的分割图核验。这样不会沿用仅覆盖书本图案、瓶身标签等局部区域的旧稀疏分割作为最终检查依据。原付费输入、provider 缓存和旧对齐候选保留；新的分割、相机或参考几何会改变混合对齐缓存指纹。此步骤不重新请求 Meshy；生成形状本身缺失的结构可能仍无法通过检查。

本机首次使用可配置一次密钥文件路径（不调用 API，也不复制密钥内容）：

```bash
.venv/bin/packing-assets configure-meshy --key-file /path/to/meshy-key.txt
```

路径保存在用户配置目录 `~/.config/packing-assets/meshy.json`（支持 `XDG_CONFIG_HOME`）。之后可省略 `--key-file`。优先级为显式 `--key-file`、`MESHY_API_KEY_FILE`、`MESHY_API_KEY`、本机保存路径。密钥文件移动后重新配置。

Meshy 批次在昂贵重建前检查凭据是否存在且可读；完全下载好的缓存不要求凭据。这个本地预检查不验证服务端余额或密钥权限，服务端问题仍会在请求时报告。默认本地分支不要求密钥。

```bash
# 默认本地重建
./run-videos.sh input output_me --backend local

# 使用同一批次的已有米制缓存，切换最终生成候选
# 会替换 output_me/models 中的本次交付；原本地重建保留在 reconstruction/
./run-videos.sh input output_me --backend meshy --key-file /path/to/meshy-key.txt

# 已配置过本机路径时
./run-videos.sh input output_me --backend meshy

# 默认快速前端，不做稠密重建、不生成平滑版；可选 Blender 展示
./run-videos.sh input output_me --backend meshy --with-blender

# 特征太少／快速定位失败时，可显式改用原完整前端
./run-videos.sh input output_me --backend meshy --meshy-frontend full

# 确实需要第二份平滑几何时再开启
./run-videos.sh input output_me --backend meshy --with-smooth

# 混合批次：例如瓶子走 Meshy，杯碗仍走本地
./run-videos.sh input output_me examples/settings-meshy.json
```

settings 文件名需与实际输入一致。`objects.<视频名>.backend` 优先于 `--backend`，支持 `local`、`meshy`。Meshy 默认 `--meshy-frontend hybrid`，固定采用 fast 的 48 帧相机设置；`--quality` 仍控制 local／完整前端，不是 Meshy 档位。输入、参照板或同一路线的设置改变时必须另用输出目录。需要保留两份交付时，先复制旧的 `scene-package.zip`。

快速路线缓存位于 `reconstruction-meshy-fast/`、`meshy-fast/`，完整路线仍在 `reconstruction/`、`meshy/`。切换路线不会覆盖另一条路线的重建。已有完整路线的付费生成，会在原视频／参照板内容指纹核对一致后复制原提交图片及生成缓存给快速路线，**不重复付费生成**；`generation-reuse.json` 记录来源，新分割只用于定标与检查。复制复用不意味着向 API 提交了新分割图片。无下载缓存时才生成新任务；输入改变会拒绝混用旧结果。

也可对已有的单物体重建单独运行，不改批次原输出：

```bash
.venv/bin/packing-assets meshy \
  --result output_me/reconstruction/<物体ID> \
  --out outputs/meshy-case \
  --key-file /path/to/meshy-key.txt --progress plain
```

`scripts/run_meshy_case.py` 保留为上述命令的兼容入口。单例最终文件在 `outputs/meshy-case/delivery/models/`。只准备图片、不调用 API：

```bash
.venv/bin/python scripts/prepare_meshy_views.py \
  --result output_me/reconstruction/<物体ID> --out outputs/meshy-inputs
```

默认不使用 Blender 或 PyBullet，保留低面数和水密检查。**Meshy 默认不做平滑、不导出 `_smooth.obj`，也不渲染第二套平滑场景**；`--with-smooth` 可恢复。local 原有平滑输出不变。需要展示时加 `--with-blender`；`--no-blender` 保留兼容。另支持 `--progress auto|plain|off`、`--verbose` 和 `--json`。已有下载缓存可不提供密钥；未完成的任务需凭据续查。也支持环境变量 `MESHY_API_KEY_FILE` 或 `MESHY_API_KEY`，推荐仓库外的单行密钥文件。密钥内容不写入任务记录或程序包。

## 当前流程与执行状态

可选快速路线（`--meshy-frontend fast`）：视频选帧 → 稀疏相机与 A4 定标 → 稀疏点引导 SAM 分割 → 四视图 Meshy → 稀疏尺度初值与多视图轮廓校正 → 低面数 OBJ → 可选展示／碰撞输出。正常情况下跳过稠密深度与融合；若稀疏前景点不足，自动复用相机和 A4 定标，补算深度及分割种子。两种情况下均跳过本地最终表面恢复。内部稀疏凸包只初始化尺度和姿态，**不会作为最终几何或碰撞体交付**。最终外形仍来自 Meshy。

批处理按物体完成整条链路，每件就绪就立即发布模型，不再等待全部本地重建完成才调用 Meshy。Meshy 成功生成但形状检查未通过的闭合模型，也会导出普通／平滑 OBJ，状态为 `retained_unvalidated`；`models/<视频名>/status.json` 保留原始 `quality_check`，不会把拒绝改成通过。此类模型只保留视觉几何，启用碰撞导出时也不生成其 URDF；批次含此类模型时不运行整体物理测试。Blender 展示标注 `SHAPE CHECK FAILED - RETAINED`。汇总场景和 ZIP 在末尾统一生成。

原始生成始终保存在 `meshy/<物体ID>/provider/source.glb`，下载后另转存 `retained/raw.obj` 供备用；即使随后定标／导出失败仍保留。**原始 OBJ 为 Y 向上、服务端未定标尺度，不能直接当米使用。** 米制候选仍在 `metric/`、`photo-aligned/`，检查报告不改写。非水密或不可解析的生成不会冒充可用碰撞体，只保留原始文件及错误说明。批次报告的 `retained_unvalidated` 单独计数；导出件数不等于形状检查合格件数。

| 部分 | 当前状态 |
|---|---|
| 默认本地视频批处理 | 已集成，不调用商业 API |
| Meshy 主入口和逐物体切换 | 已集成；逐件失败隔离，不静默回退本地模型 |
| 选帧、抠图、生成、配准、普通和平滑 OBJ | 已跑通虚拟泵头瓶；可缓存续跑 |
| 执行状态 | 已接入 `progress.json`、`run.log`、逐件与批次报告 |
| 缩短视频到成品总耗时 | 可选稀疏快速前端；`--meshy-frontend full` 保留完整路线 |
| 杯碗内腔 | Meshy 分支尚未验证；对 `open_container` 或已识别内腔的参考，在上传前拒绝，请显式用 `local` |
| Meshy 几何的 PyBullet 动力学 | 可走共用碰撞导出入口，但本次泵头瓶仅验证 OBJ 几何，未验证动力学 |
| 开源内容 | 代码、测试、文档、匿名合成基准摘要已打包；私人素材、API 密钥和生成缓存不打包 |

每个 Meshy 物体在 `meshy-fast/<物体ID>/case-report.json`（完整路线为 `meshy/`）保存阶段、状态、缓存阶段、选中候选与本次耗时；`provider/meshy-task.json` 保存可续查的任务 ID。批次 `run-report.json` 的 `object_backends` 记录逐件选择，`meshy_cases` 记录 Meshy 检查信息，失败见 `errors` 与对应 case report。前端失败不会提交新 Meshy 任务。

进度区分选图、付费提交、排队／生成、下载、米制定标、照片校正、导出。只有服务返回进度时才显示服务百分比；它不是整段视频完成率，也不预测剩余时间。其他阶段显示实际计时与缓存标志。

相同命令可重跑，保存的 API 任务会续查，已下载结果不会重新提交。输入图片、分割或相机等改变会拒绝复用。提交响应不明时停止自动重提，需先在服务端找到已有任务号再恢复，避免重复计费。并发使用同一输出目录会被锁阻止。

## 真值验证结果

2026-09-24 合成泵头瓶：真值尺寸 93 × 56 × 222 mm。Meshy 只收到四张抠出的 RGB 图。定标与配准用本地照片重建和 A4 观测，未读取真值；最后评估仅做底面中心平移，没有向真值拟合旋转或缩放。

| 最终 8,000 面 OBJ | XYZ 尺寸绝对误差 mm | 表面 P95 mm | 体积 IoU |
|---|---|---:|---:|
| 本地标准重建 | 0.13 / 0.75 / 0.84 | 0.93 | 97.67% |
| 本地平滑版 | 0.19 / 0.35 / 1.16 | 0.86 | 98.13% |
| Meshy | 2.41 / 0.10 / 1.30 | 1.69 | 94.75% |
| Meshy 平滑版 | 2.56 / 0.17 / 1.59 | 1.73 | 94.61% |

四个导出模型均经重新读取和 Blender BMesh 独立检查：边界边、非流形边为零，封闭体积。均通过相同门槛：尺寸误差 ≤ max(3 mm, 3%)、表面 P95 ≤ 3 mm、体积 IoU ≥ 90%。表面采用双向各 100,000 点采样，体积采用 1.5 mm 抖动网格；不能把采样 P95 当作最大误差。

Meshy 保留了泵头与瓶身，但瓶肩多出不真实的凹凸。本例本地标准重建更准确。单个合成案例不能证明真实物品、隐藏面或内腔质量。公开数值摘要见 [benchmark JSON](benchmarks/meshy-pump.json)；开发机器的原始结果保留在忽略的 `outputs/meshy-virtual-comparison/`。

本例 Meshy 7.1 standard 无纹理任务实测 **25.95 秒、20 credits**（生成、轮询、下载）。这是完整路线的历史实测，不是固定时间／价格承诺。该本地标准前端记录约 1051 秒，其中稠密深度约 977 秒；**不能将 26 秒写成视频到成品的总耗时**。快速路线在稀疏点充足时不执行稠密深度；默认混合路线会使用 fast 档深度测量。

### 真实视频耗时与快速前端方向

2026-09-24，现有三个真实视频 fast 前端记录如下（历史阶段累计；不是本次缓存重跑计时）：

| 视频 | 本地前端累计秒 | 其中稠密重建秒 | Meshy 生成／轮询／下载秒 |
|---|---:|---:|---:|
| 盒子 | 177.61 | 93.95 | 36.03 |
| 键盘 | 196.04 | 94.52 | 25.75 |
| 飞机 | 200.35 | 108.06 | 25.40 |

同日快速前端从三个原视频重新处理，未复用旧稠密深度或分割；键盘随后复用该新前端做最终配准。最终几何复用同视频已有 Meshy GLB，本次没有新增付费生成。新选出的四张抠图另行生成并检查，但没有再次提交 API。

| 视频 | 快速前端实测秒 | 新配准／照片校正秒 | 相比旧路线最大外尺寸变化 mm | 预计首次总耗时秒 |
|---|---:|---:|---:|---:|
| 盒子 | 84.30 | 5.86 | 0.25 | 129 |
| 键盘 | 88.05 | 17.41 | 2.35 | 145 |
| 飞机 | 88.44 | 7.42 | 2.66 | 125 |

前端耗时减少约 55%。最后一列是新前端／配准／导出实测加上历史 API 延迟的**估算**，不含可选 Blender 和整批打包，不是一次新的全程付费调用实测，也不保证 API 排队时间。尺寸变化是两条估计路线之间的一致性，**不是独立真实尺寸误差**。盒子、键盘通过原照片检查；飞机仍低于门槛，按保留策略导出。数值及分阶段记录见 [快速路线基准](benchmarks/meshy-fast.json)。

旧流程主要耗时在本地完整重建，不只是抽帧准备。快速路线已将深度分割种子替换为 A4 米制稀疏特征，保留多视图校正，避免直接用纸张／物体像素比例推尺寸。它假设一件主要物体放在平桌面、有足够纹理和高于桌面的可辨识特征；极薄、无纹理、透明／反光物体可能缺少可靠的稀疏点，会自动补算深度种子后继续 Meshy，不需要重新上传或手动改为 full。补救需要本机 CUDA 深度依赖，会增加耗时；补救仍无法定位物体则明确失败，不提交付费任务。`reconstruction.json` 的 `frontend_fallback` 记录原因，`frontend_prompt_source` 记录实际种子来源。

## 尺度与质量边界

Meshy 输出先从 Y-up 转成 Z-up，再统一缩放、旋转、平移到本地 A4 米制参考，随后用 12 张标定照片优化姿态与统一尺度、用全部有效视图检查轮廓。禁止按 XYZ 分别拉伸，不使用 AI 自动尺寸估计替代 A4。参考模型和分割也会有误，照片检查不是独立真值；无法恢复被生成模型理解错的形状。

默认参数为 `meshy-7.1`、`standard`，关闭纹理、重拓扑、图像增强和 `auto_size`。后处理共用现有降面、受限平滑和封闭性检查，不用全局凸包隐藏质量问题。

接口资料：[多图 API](https://docs.meshy.ai/en/api/multi-image-to-3d)、[计费](https://docs.meshy.ai/en/api/pricing)、[官方 MCP](https://github.com/meshy-dev/meshy-mcp-server)。Meshy 是独立商业服务，本项目 MIT 许可不覆盖其服务或生成内容条款。


## 尺寸校正版

Meshy 分支现在自动尝试额外的 `物体名_dimensions.obj`，原 `物体名.obj` 保留不变，默认场景及碰撞体仍用原版本。不额外调用 Meshy，也不做平滑。校正版继承原低面数网格；例如瓶子仍为 8,000 面。

- 如果同批次已有本地重建，或手机队列中已有**同一视频内容**的已完成本地任务，会核对视频 SHA-256、参照板和重建指纹，再使用通过检查的本地最终模型尺寸作为约束。文件名可以不同。不会拿稀疏点包围盒当真实尺寸。
- 没有可用本地结果时，采用 A4 相机定标下的多视角轮廓，受限调整各方向尺寸；水平主轴不明确的近圆／近方物体保持两个水平缩放相同。
- 调整限制在约 ±10%，同时比较面积重合、边缘和轮廓范围。预留视角不用于此次尺寸拟合，也排除提交 Meshy 的图片；但相机、分割、之前的候选选择共享数据，因此不属于独立物理精度验证。
- 只有预留视角有明确改善且原水密／轮廓门槛仍通过时，才额外发布校正版。失败或改善不明确时保留原模型，原因见 `models/物体名/dimension-report.json` 和 case report。
- 低面数导出可能造成小于高分辨率候选的尺寸差异；报告同时记录候选和最终导出尺寸。校正中间模型留在 `meshy-fast/物体ID/dimension-aligned/`，未通过的中间模型不是交付结果。

本地结果在 Meshy 任务完成之后才出现时，需用原输出目录重新运行，新的尺寸参考会使尺寸校正缓存更新；已有 API 生成仍复用。混合模式现在会自动补齐本地尺寸参考；快速模式只复用已有本地结果。

同视频瓶子：原 Meshy 候选约 72.2 × 72.5 × 180.7 mm，本地候选约 68.9 × 68.0 × 190.3 mm。约束后候选尺寸与本地一致，预留视角平均 IoU 从 94.68% 提高到 95.81%，本地尺寸校正约 3.3 秒；盒子、键盘改善不明确，未发布校正版。实测记录见 [尺寸校正对照](benchmarks/meshy-dimensions.json)。这些是模型间对比，不代表真实尺寸误差。

已知真值的合成盒子回归测试：160 × 90 × 100 mm 的真值，故意变形成 168.8 × 86.4 × 95 mm，照片校正恢复至约 160.0 × 90.1 × 99.8 mm。它验证理想相机／分割下算法能恢复比例，不保证实拍具有同等精度。


## 默认混合模式（hybrid）

```bash
# 默认：一次输入，本地测量尺寸 + Meshy 外形
./run-videos.sh input output_me --backend meshy
# 可选：更快的照片校正，不主动补建本地测量模型
./run-videos.sh input output_me --backend meshy --meshy-frontend fast
```

混合和快速模式共用 `reconstruction-meshy-fast/` 与 `meshy-fast/`，可在同一输出目录切换，保持已付费 Meshy 任务及其原始输入。首次混合模式直接准备深度引导的分割；已有快速结果保持原分割不变，在独立的 `local-measurement/` 工作区复用**已完成**的相机、定标、深度、分割阶段，补建测量候选，不重复抽帧或估计相机。独立文件副本避免本地重建升级或恢复过程改写已提交图片。未完成的深度缓存不作为成功检查点复用。

找到同视频、同参照板的有效本地候选时直接使用；没有时自动构建。尺寸约束允许围绕本地参考最多约 0.5% 微调，以兼顾重建离散误差和照片边缘，校验门槛不变。输出仍为原 OBJ 加可选的 `_dimensions.obj`；默认 Blender 场景和碰撞模型仍使用原 OBJ。尺寸候选失败时不冒充通过，原因记录于报告；本地测量阶段失败时保留照片校正路径，不丢弃 Meshy 备用结果。混合模式需要本地 CUDA 重建依赖；不配置这些依赖时请显式选 fast（仅在稀疏前景足够时无需深度）。

`hybrid-measurement.json` 记录尺寸参考和复用阶段，case report 的 `hybrid_measurement` 同步汇总。2026-09-24 隔离实测：没有独立本地版本、已有深度/分割/API 缓存的瓶子，自动补建本地测量约 10.9 秒，额外尺寸拟合约 6.2 秒，成功输出 8,000 面校正版。此数据**不包含首次抽帧、深度重建或云端生成耗时**，不能作为首次完整耗时承诺。
