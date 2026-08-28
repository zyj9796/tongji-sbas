# 天津论文严格复现

本目录用于从原始 BC3 条带模式 SLC、天津建筑轮廓和 SRTM DEM 重新复现论文《融合 InSAR 与建筑物轮廓矢量的城市三维重建》。它与既有同济校区实验完全隔离，不复用同济校区的配准结果、投影缓存、先验高度或建筑高度产品。

当前状态：**完整处理链已运行并通过最终结构检查（2026-08-25）**。已完成 21 景条带拼接/裁剪与配准、48 对 GAMMA 差分干涉、建筑独立 GAMMA-MCF 解缠、SBAS 两参数反演、建筑级汇总、全量矢量回填和六张中文 SVG。数值结果是可复现的实验输出，不等同于论文缺少原始 LiDAR 对照时所声称的外部精度。

## 输入

- 原始 SLC：`/home/u/Downloads/中国天津市点位2`
- 建筑矢量：`../天津建筑轮廓数据/天津.shp`
- 论文：`../融合InSAR与建筑物轮廓矢量的城市三维重建.pdf`
- 原始 MATLAB：`../my_study/`
- DEM：SRTM 1 Arc-Second Global，瓦片 `N39E117`，下载后记录来源、校验和、水平/垂直基准。

## 不可变规则

1. 主方法以论文正文为准；原始 MATLAB 只用于恢复正文未写出的运行细节。
2. 建筑 `Floor` 字段可按论文用于离散属性栅格和对象关联，不得换算成高度后填充、筛选或校正 InSAR 结果。
3. 外部 DEM 只提供地形相位和地面绝对高程基准；建筑高度必须来自解缠相位的时序反演与建筑级稳健聚合。
4. 论文存在互相冲突的描述时，主结果采用与论文图表及统计量相符的分支，同时保留另一分支的敏感性结果。
5. 每一步必须保留输入清单、命令、参数、日志、校验和和质量指标；失败结果不得被先验值替换。

## 正式输出

- 全量建筑矢量：`results/paper_strict/building_height_final.gpkg`
- 有解建筑表：`results/paper_strict/building_height_final.csv`
- 像元级 SBAS 解：`results/paper_strict/pixel_height_independent_mcf_fixed_far_48.npz`
- 后验楼层对照（不回写结果）：`results/paper_strict/floor_prior_posthoc_audit.csv`
- 六张独立中文 SVG：`results/figures/`
- 通俗版 Word 报告：`results/report/天津建筑三维重建论文复现总结报告_通俗版.docx`
- 最终机器检查：`inventory/final_reproduction_validation.json`

完整复现总结见 [`docs/05_天津建筑三维重建论文复现总结报告.md`](docs/05_天津建筑三维重建论文复现总结报告.md)。详细冻结方案见 [`docs/00_论文与数据审计及冻结方案.md`](docs/00_论文与数据审计及冻结方案.md)，执行结果、论文/代码差异和限制见 [`docs/01_严格复现执行结果与审计.md`](docs/01_严格复现执行结果与审计.md)。
