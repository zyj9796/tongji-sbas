# 当前文献路线、已完成工作与后续优化计划

Last updated: 2026-07-06

## 1. 文献方法基线

依据 `文献和原始代码/融合InSAR与建筑物轮廓矢量的城市三维重建.pdf`，当前复现应围绕以下闭环，而不是只做普通 InSAR 栅格测高：

1. 二维建筑物轮廓矢量前置进入 InSAR 处理流程。
2. 将 WGS84/平面建筑轮廓映射到 SAR 雷达坐标，生成建筑目标 mask。
3. 通过连通域/聚类提取拓扑独立建筑孤岛，避免跨建筑边界传播解缠误差。
4. 在建筑孤岛内部进行独立相位解缠，抑制全局 MCF 在高程突变边界处的整周误差扩散。
5. 在 SBAS 时序框架下，以多基线差分干涉相位为观测量，联合估计 DEM residual 和形变项。
6. 将像元级 residual 转为屋顶高程证据，再在建筑单体内进行稳健聚合。
7. 将反演高度挂接回二维建筑轮廓，用于 LOD1 建筑白模或矢量制图。

当前项目的主线必须坚持：建筑矢量 `height` 字段不参与拟合、筛选、填补、标定或 QC；严格 InSAR 无解建筑保留空值。

## 2. 当前采用的主线结果

当前推荐采用 `cleanid_redshift_audited` 分支：

- 建筑底图：clean equal-height 建筑矢量。
- SAR 几何：clean roof-only 投影。
- 岛分割：按 `clean_id` 再拆分连通域，消除多建筑混岛。
- 局部投影修正：只对 red/review 建筑做 SAR 幅度/边缘位移搜索，并且只接受 InSAR 内部指标确认有收益的 16 个 clean_id。
- 解缠：建筑孤岛内 paper-like 独立解缠。
- 反演：SBAS/LGR DEM residual，`height = DSM_RDC + residual - 4 m`。
- 像元门限：coherence `>=0.75`，amplitude dispersion `<=0.40`，valid pairs `>=12`，LGR RMSE `<=1.25 rad`，Bperp span `>=120 m`。
- 顶面选择：从最高点向下做 one-sided Grubbs 检验，不使用 p95/p90 作为最终 fallback。

当前最终统计：

- 总建筑：1028
- 严格 InSAR 解：239
- 无严格解：789
- reliability：high 168，medium 42，review 29，no_solution 789
- 顶面高度中位数：22.55 m
- 顶面高度 P05/P95：3.70 / 48.99 m
- 相对 clean_id split 分支：解算数量 239 -> 239，review 34 -> 29，high 158 -> 168，lost 0

## 3. 当前保留的关键文件

核心输入：

- `configs/tongji_reproduction.json`
- `data/shp/clean_equal_height/tongji_clip_rslc_extent_equal_height_clean.geojson`
- `work/projection/20200708_clean_equal_height_roof_projection_sar.geojson`
- `work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128.npy`
- `work/masks/building_fid_mask_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited.npy`
- `work/masks/island_label_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.npy`
- `work/masks/islands_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_split.csv`
- `work/baselines/reselected_triangular_22_gamma_bperp_paircoh0.39.csv`
- `work/gamma_sbas/dem/20200708_dsm_rdc.hgt`
- `work/mli/amplitude_dispersion_crop_bmp.npy`
- `work/mli/mean_crop_bmp_amplitude.npy`

核心中间结果：

- `work/height/island_pixel_lgr_heights_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv`
- `work/height/height_points_clean_equal_height_roof_only_full_area_128_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120.csv`
- `work/projection/cleanid_redshift_audited_accept_clean_ids.csv`
- `work/projection/cleanid_split_red_building_mask_shift_metrics_audited.csv`

最终产品：

- `results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.geojson`
- `results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_insar_only.csv`
- `results/metadata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_topdown_grubbs_summary.json`

当前临时只输出一个关键图件：

- `results/pic_all/svg/current_strict_clean_equal_height_full/203_cleanid_redshift_audited_building_topstats_height.svg`

## 4. 已完成的优化

1. Clean equal-height 建筑矢量清洗，减少相邻同高建筑导致的投影和聚合混淆。
2. Clean roof-only SAR 投影与 full-area 128 范围 mask 建立。
3. 文献路线 LGR 反演门限收敛到 `coh075_DA040_minp12_rmse125_bspan120`。
4. 最终高度选择从 p95/p95-floor 改为最高点向下 Grubbs 检验。
5. clean_id split 岛分割将多 clean_id 混岛从 115 个降为 0 个，解算建筑从 205 提升到 239。
6. 对 red/review 建筑做投影偏差搜索；全量 redshift 会丢 3 栋解，因此不采用。
7. audited redshift 只保留 16 个 InSAR 内部收益为正的位移，最终 review 从 34 降到 29，且 lost 为 0。

## 5. 已废弃或降级为诊断的分支

以下分支不再作为当前主线产品保留，只保留其结论在本文档中：

- DEM0/no-HGT 分支：方法测试，结果高度中位数明显偏低，不替代 DSM residual 主线。
- APS/deformation 分支：用于残差和形变诊断，不是当前建筑高度主线。
- p90/p95/likely_top/p95-floor 分支：已被最高点向下 Grubbs 取代。
- full redshift 分支：review 下降但丢失 3 栋严格解，不采用。
- 早期 `paper_opt_coh080_DA035` 严格分支：质量更严但覆盖不足，作为历史敏感性结论即可。

## 6. 后续优化计划

优先级 1：审计剩余 29 栋 review 建筑。

- 按 review 原因分为：顶端 Grubbs 剔除过多、局部投影偏差、建筑内部多散射体混叠、低相干/高 DA、LGR 模型 RMSE 不稳。
- 输出逐栋表和诊断图，只用 InSAR 内部指标和 SAR 幅度/边缘证据，不用建筑 `height` 字段。
- 目标是把 review 分为可修正、应保留 review、应转无解三类。

优先级 2：对 789 栋无解建筑做失败原因分解。

- 统计无解来自：无 SAR roof mask、mask 像元太少、coherence 不足、DA 超限、valid pairs 不足、Bperp span 不足、RMSE 超限、解缠失败。
- 这一步决定后续是优化投影、优化门限，还是需要多轨道/更多干涉对。

优先级 3：只针对明确失败类型做门限敏感性。

- 固定当前 audited mask 和 top-down Grubbs。
- 小范围测试：coh `0.75 -> 0.70`，DA `0.40 -> 0.45`，min pairs `12 -> 10`，RMSE `1.25 -> 1.50`。
- 每组必须报告新增解、丢失解、review 增量、median RMSE、空间变化和新增建筑的 QC 分类。

优先级 4：扩展投影修正到“无解但 SAR 证据强”的建筑。

- 只对无解建筑中存在清晰 SAR 亮脊/边缘但 mask 偏移的对象做局部搜索。
- 接受条件必须是新增严格 InSAR 解或显著改善 LGR 内部指标，不能仅凭幅度评分。

优先级 5：补齐 LOD1 白模交付。

- 用最终 `height_insar_m` 挂接 clean 建筑矢量。
- 严格无解建筑不拉伸或单独标记。
- 输出 GeoJSON/OBJ/Blender 可导入格式，并附带高度来源和 reliability 字段。

## 7. 文件整理原则

保留：

- 原始数据、论文和外部参考。
- 当前主线的输入、中间结果、最终矢量/表格/摘要/图件。
- 能重建当前主线的代码和 README。

删除：

- 可重建缓存：`tmp/uv-cache`、PDF 临时渲染页、`__pycache__`。
- 已被当前主线替代的旧 GeoJSON/CSV/metadata/figures。
- 与当前建筑高度主线无关的 deformation/APS 大型实验输出。
- 未采用的 full redshift、DEM0、p95/p90/likely_top 结果产物。
