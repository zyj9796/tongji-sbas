# ZJC 原代码主导的同济高层建筑复现

本目录把 `zjc/my_study` 的三个核心 MATLAB 模块按原数值逻辑迁移到同济数据接口：

- `unwrap_phase_matrix.m`：9 点局部邻域、4 邻接 KNN、双线性相位差模型、Bisquare 抗差估计和 LSMR 图积分；
- `local_knn.m`：KNN 与 Delaunay 弧；
- `LGR_demerror_est.m`：垂直基线/时间基线联合的非加权逐像元 LGR；
- 建筑高度：按原脚本取 `max(DEM error)-min(DEM error)`。

只修正原代码中会造成同济数据不可运行或静默错误的接口问题：背景像元不再进入图、按主从日期连接干涉对、零相位不再表示 NoData、有效干涉对数显式设为 12。没有引入当前工程的 Grubbs、p95-floor、投影位移优化高度选择规则。

## 运行

```bash
env MPLCONFIGDIR=/tmp/mpl-zjc-original \
.venv/bin/python code/reproduce_zjc_tongji/run_zjc_original_core.py

env MPLCONFIGDIR=/tmp/mpl-zjc-original \
.venv/bin/python code/reproduce_zjc_tongji/build_gamma_corrected_projection.py

env MPLCONFIGDIR=/tmp/mpl-zjc-original \
.venv/bin/python code/reproduce_zjc_tongji/generate_paper_svgs.py
```

数值缓存和审计表放在 `work/zjc_original_reproduction/`。用户指定的最终目录 `picall/` 只写 SVG。

## 图件与论文对应关系

每个画面均单独输出，不再把多个子图合并到一张图中；图名和图内可见文字均为中文。

| 输出 | 论文对应内容 | 同济复现内容 |
|---|---|---|
| `01_同济建筑轮廓与严格有解建筑.svg` | 研究区 | 同济全部建筑与260栋严格SBAS有解建筑 |
| `02_同济雷达平均幅度.svg` | 研究区雷达影像 | 同济 SAR 平均幅度 |
| `03_干涉对时空基线网络.svg` | 时空基线网络 | 选中干涉对及获取日期 |
| `04`–`09_建筑重叠示例*.svg` | 建筑重叠处理 | 三个示例各自的排除前、排除后画面 |
| `10_建筑孤岛全景.svg` | 建筑孤岛提取 | 最新GAMMA+SAR校正投影形成的连续独立屋顶岛；跨楼建筑体冲突、低可靠证据岛和过小岛已剔除，最终高度有解与否不参与选择 |
| `11_建筑孤岛局部放大.svg` | 建筑孤岛提取 | 屋顶核心区与SAR亮目标局部细节 |
| `12_屋顶核心区SBAS高度图.svg` | 高度反演 | 稳定地面定标、多初值模糊度搜索后的屋顶核心像元高度 |
| `13_全局解缠高度图.svg` | 高度反演对照 | 常规全局解缠结果 |
| `14_严格有解建筑三维重建.svg` | 三维重建 | 仅260栋严格有解建筑的LOD1重建 |
| `15_投影初始化先验高度分布.svg` | 投影初始化 | 只用于第一次GAMMA屋顶投影的矢量先验高度 |
| `16_全局解缠高度分布.svg` | 高度空间分布 | 全局解缠高度 |
| `17_屋顶核心区SBAS建筑高度分布.svg` | 高度空间分布 | 260栋严格屋顶SBAS高度全部标注数值（m）；768栋无解不填充 |
| `18_全部建筑高度辅助投影至雷达影像.svg` | 建筑雷达投影 | 仅显示GAMMA屋顶投影；局部窗口按屋顶尺寸和建筑高度自适应为±3–9像元，以20200708单景搜索、非参考时相验证 |

按用户要求，不生成论文 Fig. 8–12 对应的误差散点、直方图、残差、箱线图和 CDF。最终保留 18 张独立 SVG，不输出 PNG、PDF 等其他格式。

## 解释边界

- 论文试验选择 79 栋 30 层以上建筑；同济只有 2 栋超过 30 层，因此默认用 `Floor>10` 得到数量接近的高层类比子集。
- 图18读取 `data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.gpkg` 的1028栋建筑。矢量 `height` 只定义地面4 m到屋顶 `4 m + height` 的几何拉伸量；底面和屋顶顶点均由GAMMA `coord_to_sarpix` 投影，并使用 `DIFF_par` 改正。后续SAR残差校正只使用按日期奇偶拆分的训练/验证幅度与边缘证据，绝不使用 `height` 作为拟合、校正、筛选、填充或质量控制目标。
- `Floor`/矢量 `height` 可用于高层类比子集元数据和图15先验分布；ZJC及全局反演高度始终由相位解缠后的DEM误差极差计算，缺失反演结果保持缺失，禁止以 `height` 填充或替换。
- GAMMA校正缓存为 `work/zjc_original_reproduction/20200708_all_building_gamma_corrected_projection_sar.geojson`；逐栋校正指标和审计摘要分别为 `20200708_all_building_gamma_projection_metrics.csv` 与 `20200708_all_building_gamma_projection_summary.json`。
- 本地没有论文使用的独立 LiDAR 真值，也没有 CNBH-10m/GBA 产品。因此图 7–11 不能解释为精度验证，图 12 的第一列是矢量先验而不是 CNBH/GBA。
- 常规对照使用整幅相位连续域解缠，不冒充论文中的 MCF 软件实现。
