# 当前文献路线、已完成工作与后续优化计划

Last updated: 2026-07-07

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
8. 2026-07-07 完成剩余 29 栋 review 建筑逐栋审计，只使用 InSAR 内部指标和 SAR 幅度/边缘证据，不使用建筑矢量 `height` 字段。

Review 审计结果：

- 29 栋 review 全部属于 `max_reject_use_top_down_grubbs_many_removed`：最高点假设被 Grubbs 拒绝，且顶端向下剔除超过 2 个像元后才得到保留顶面。
- 可控投影偏差候选 3 栋：`clean_id` 344、576、600。后续若扩展投影修正，应先只针对这 3 栋做小范围复核，且接受条件必须是严格 InSAR/reliability 改善。
- 继续保留 review 26 栋：主要原因是建筑内部多散射体/上尾混叠、LGR RMSE 接近上限、质量门限附近、或已接受投影修正但顶端仍不稳定。
- 保守自动转无解候选 0 栋：目前没有 review 建筑满足多项内部弱支撑条件，不能自动改为 `no_solution`。
- 审计文件：`docs/review_building_audit_20260707.csv`、`docs/review_building_audit_20260707.md`、`docs/review_building_audit_20260707.svg`、`docs/review_building_audit_20260707_summary.json`。

789 栋 no-solution 建筑现有证据分解：

- 无 SAR roof island：383 栋。这一类优先做投影/mask 与 SAR 亮脊证据筛查，不应直接进入门限放宽。
- DA `<=0.40` 后不足 20 像元：27 栋。这一类可作为后续 `DA 0.40 -> 0.45` 小范围敏感性候选，但仍需报告新增解质量。
- LGR valid-pairs/unwrap 覆盖不足：355 栋。诊断版 LGR 显示这些建筑在 coherence 筛选和建筑孤岛内解缠后，达到 `min_pairs>=12` 的像元不足 20。
- LGR RMSE 超限：24 栋。这些建筑通过 valid-pairs 与 Bperp 支撑后，在 `RMSE<=1.25 rad` 后不足 20 像元。
- 当前诊断中没有 Bperp span 主导的 no-solution 建筑。
- 审计文件：`docs/no_solution_failure_audit_20260707.csv`、`docs/no_solution_failure_audit_20260707.md`、`docs/no_solution_failure_audit_20260707.svg`、`docs/no_solution_failure_audit_20260707_summary.json`、`docs/lgr_failure_gate_diagnostics_buildings_20260707.csv`、`docs/lgr_failure_gate_diagnostics_islands_20260707.csv`、`docs/lgr_failure_gate_diagnostics_20260707_summary.json`。
- 下一步敏感性优先级：先筛查 383 栋无 SAR roof island 的投影/SAR 证据，再针对 355 栋做 `min_pairs 12 -> 10` 小范围测试，针对 24 栋做 `RMSE 1.25 -> 1.50` 小范围测试，最后对 27 栋做 `DA 0.40 -> 0.45` 小范围测试。

门限敏感性预筛结果：

- `min_pairs 12 -> 10`：355 栋 valid-pairs/unwrap 失败建筑中，26 栋可达到 20 个 LGR 像元，值得做一次产品级小范围重建和基线对比。
- `RMSE 1.25 -> 1.50 rad`：24 栋 RMSE 失败建筑中，12 栋可达到 20 个 LGR 像元，比例最高，但会直接放宽模型拟合门限，必须重点检查 residual 和空间聚集。
- `DA 0.40 -> 0.45`：27 栋 DA 限制建筑中，0 栋达到 20 个 LGR 像元，当前不应优先做 DA 放宽。
- 候选文件：`docs/no_solution_gate_sensitivity_buildings_20260707.csv`、`docs/no_solution_gate_sensitivity_islands_20260707.csv`、`docs/no_solution_gate_sensitivity_20260707.md`、`docs/no_solution_gate_sensitivity_20260707.svg`、`docs/no_solution_gate_sensitivity_20260707_summary.json`。

`min_pairs=10` 产品级小测试：

- 在保持 audited mask、coh `>=0.75`、DA `<=0.40`、RMSE `<=1.25 rad`、Bperp span `>=120 m`、top-down Grubbs 不变时，严格有解建筑从 239 增至 276，新增 37 栋，丢失 0 栋。
- 可靠性变为 high 182、medium 57、review 37、no_solution 752；新增 37 栋中 high 27、medium 9、review 1。
- 当前已解建筑中，`min_pairs=10` 与主线高度差的中位数为 0.00 m，P05/P95 为 0.00/8.90 m。
- 结论：`min_pairs=10` 可以作为候选分支，但 review 数从 29 增至 37，不能直接替代当前主线；新增建筑需继续做内部 residual、顶端稳定性、空间聚集和 SAR/mask 一致性审计。
- 文件：`docs/minpairs10_pixel_lgr_summary_20260707.json`、`docs/minpairs10_topdown_grubbs_summary_20260707.json`、`docs/minpairs10_vs_current_topdown_comparison_20260707.csv`、`docs/minpairs10_product_test_20260707.md`、`docs/minpairs10_product_test_20260707.svg`、`docs/minpairs10_product_test_20260707_summary.json`。

`RMSE=1.50 rad` 产品级小测试：

- 在保持 audited mask、coh `>=0.75`、DA `<=0.40`、min pairs `>=12`、Bperp span `>=120 m`、top-down Grubbs 不变时，严格有解建筑从 239 增至 251，新增 12 栋，丢失 0 栋。
- 可靠性变为 high 179、medium 45、review 27、no_solution 777；新增 12 栋中 high 8、medium 3、review 1。
- 当前已解建筑中，`RMSE=1.50` 与主线高度差的中位数为 0.00 m，P05/P95 为 0.00/15.78 m，高于 `min_pairs=10` 分支的 P95 变化。
- 结论：`RMSE=1.50` 新增数量少于 `min_pairs=10`，review 数较低，但高度上尾变化更大且直接放宽模型拟合门限，应作为谨慎候选，不优先替代主线。
- 文件：`docs/rmse150_pixel_lgr_summary_20260707.json`、`docs/rmse150_topdown_grubbs_summary_20260707.json`、`docs/rmse150_vs_current_topdown_comparison_20260707.csv`、`docs/rmse150_product_test_20260707.md`、`docs/rmse150_product_test_20260707.svg`、`docs/rmse150_product_test_20260707_summary.json`。

两个候选分支综合决策：

- `min_pairs=10` 新增 37 栋，`RMSE=1.50` 新增 12 栋，二者新增重叠 8 栋。
- `min_pairs=10` 独有新增 29 栋；`RMSE=1.50` 独有新增 4 栋。
- 当前有解建筑中，阈值分支导致高度变化超过 8 m 的审计项共 37 条：`min_pairs=10` 16 条，`RMSE=1.50` 21 条。
- 决策：当前严格产品继续作为主线；`min_pairs=10` 作为高覆盖候选分支，优先审计新增 37 栋和 16 条大幅高度变化；`RMSE=1.50` 只作为 residual 审计分支，不优先替代。
- 文件：`docs/threshold_variant_added_building_decision_table_20260707.csv`、`docs/threshold_variant_large_height_change_audit_20260707.csv`、`docs/threshold_variant_decision_20260707.md`、`docs/threshold_variant_decision_20260707.svg`、`docs/threshold_variant_decision_20260707_summary.json`。

偏低修正重估：

- 用户指出当前建筑高度估计严重偏低后，新增 `p95_floor_reestimated` 产品作为修正版顶面高度。
- 这版不重跑 LGR，不放宽 coherence/DA/min-pairs/RMSE/Bperp 门限，不使用建筑矢量 `height` 字段；只把最终顶面选择从纯 `top_down_grubbs` 改为 `descending_grubbs_p95_floor`，避免 Grubbs 过度剔除真实屋顶高点。
- 有解建筑仍为 239 栋，无解仍为 789 栋；reliability 仍为 high 168、medium 42、review 29、no_solution 789。
- 高度统计从 median/P05/P95 = 22.55/3.70/48.99 m 调整为 23.27/5.22/51.00 m。
- 与旧 top-down 产品相比，239 栋可比建筑中 29 栋被抬高；25 栋抬高超过 2 m，19 栋超过 5 m，8 栋超过 10 m。
- 新产品：`results/geodata/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_p95_floor_reestimated_insar_only.geojson`、`results/tables/tongji_building_height_cleanid_redshift_audited_paper_lit_coh075_DA040_minp12_rmse125_bspan120_p95_floor_reestimated_insar_only.csv`。
- 新图件：`results/pic_all/svg/current_strict_clean_equal_height_full/204_cleanid_redshift_audited_building_height_p95_floor_reestimated.svg`。
- 差异表：`results/tables/tongji_building_height_p95_floor_reestimate_vs_topdown_grubbs_diff.csv`。

投影局部偏移校正分支：

- 用户指出当前投影与 SAR 底图局部仍有偏移后，新增独立目录 `results/projection_correction_20260707/` 保存投影校正分支。
- 当前投影、audited mask、当前 island label、当前 islands CSV、当前 audited redshift metrics 已归档到 `results/projection_correction_20260707/baseline_current/`。
- 新校正只使用 SAR 幅度/边缘证据和当前 clean_id roof mask 几何，不使用建筑矢量 `height` 字段。
- 全区搜索 678 个有足够 mask 像元的建筑，得到 187 个原始 SAR-shift 候选，最终保守接受 119 个局部位移；变更 mask 像元 19107 个，中位接受位移为 row +2、col -2 像元。
- 校正后 clean-id split islands 为 663 个，多 clean_id 混岛仍为 0，丢弃小组件像元 525 个。
- 输出：`results/projection_correction_20260707/20200708_clean_equal_height_roof_projection_sar_local_sar_corrected.geojson`、`results/projection_correction_20260707/building_fid_mask_clean_equal_height_roof_only_full_area_128_local_sar_corrected.npy`、`results/projection_correction_20260707/island_label_local_sar_corrected_cleanid_split.npy`、`results/projection_correction_20260707/islands_local_sar_corrected_cleanid_split.csv`、`results/projection_correction_20260707/local_sar_projection_shift_metrics.csv`。
- QA 图件：`results/projection_correction_20260707/figures/projection_correction_overlay.svg`、`results/projection_correction_20260707/figures/local_shift_examples_top12.svg`。
- 当前状态：这是投影校正候选分支，尚未替代 active height workflow；下一步应使用校正后 island label 重跑严格 LGR，并与当前 audited 分支比较 solved/review/lost/height stability 后再决定是否提升为主线。

## 5. 已废弃或降级为诊断的分支

以下分支不再作为当前主线产品保留，只保留其结论在本文档中：

- DEM0/no-HGT 分支：方法测试，结果高度中位数明显偏低，不替代 DSM residual 主线。
- APS/deformation 分支：用于残差和形变诊断，不是当前建筑高度主线。
- p90/p95/likely_top/p95-floor 分支：已被最高点向下 Grubbs 取代。
- full redshift 分支：review 下降但丢失 3 栋严格解，不采用。
- 早期 `paper_opt_coh080_DA035` 严格分支：质量更严但覆盖不足，作为历史敏感性结论即可。

## 6. 后续优化计划

优先级 1：审计剩余 29 栋 review 建筑。已完成，见第 4 节 2026-07-07 审计结果。

- 按 review 原因分为：顶端 Grubbs 剔除过多、局部投影偏差、建筑内部多散射体混叠、低相干/高 DA、LGR 模型 RMSE 不稳。
- 输出逐栋表和诊断图，只用 InSAR 内部指标和 SAR 幅度/边缘证据，不用建筑 `height` 字段。
- 目标是把 review 分为可修正、应保留 review、应转无解三类。

优先级 2：对 789 栋无解建筑做失败原因分解。已完成，并补充了 LGR 阶段逐门限诊断输出。

- 统计无解来自：无 SAR roof mask、mask 像元太少、coherence 不足、DA 超限、valid pairs 不足、Bperp span 不足、RMSE 超限、解缠失败。
- 这一步决定后续是优化投影、优化门限，还是需要多轨道/更多干涉对。

优先级 3：只针对明确失败类型做门限敏感性。

- 固定当前 audited mask 和 top-down Grubbs。
- 根据 2026-07-07 预筛结果，优先做产品级小范围测试：min pairs `12 -> 10`，然后 RMSE `1.25 -> 1.50`；DA `0.40 -> 0.45` 暂不优先，coh `0.75 -> 0.70` 需等待 valid-pairs 失败对象的空间/相干诊断后再决定。
- 每组必须报告新增解、丢失解、review 增量、median RMSE、空间变化和新增建筑的 QC 分类。
- 当前已完成 `min_pairs=10` 与 `RMSE=1.50` 两个产品级 docs-only 测试。下一步不是继续盲目放宽门限，而是审计两个候选分支新增建筑和高度变化较大的保留建筑。

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
