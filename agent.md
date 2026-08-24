# Agent Current State

Last updated: 2026-08-25

## Latest checkpoint — Tianjin paper strict reproduction from raw BC3 SLC

- Isolated root: `zjc/strict_reproduction/`. This branch starts from `/home/u/Downloads/中国天津市点位2`, `zjc/天津建筑轮廓数据/天津.shp`, SRTMGL1 `N39E117`, the paper, and its MATLAB; it does not reuse the Tongji-campus products described below.
- Full stack complete: 21 dates (2023-10-07 to 2024-06-05), two strip segments per date, one 10000x7000 common crop, one reference SLC plus 20 RSLCs. All coregistration fits pass the 0.1-pixel gate.
- Main paper-result network: 48 differential interferograms. Text-strict sensitivity network: 45 pairs because the paper's 48-pair result includes three 55-day pairs despite the prose saying <=44 days.
- Formal island branch preserves overlapping SAR-coordinate hypotheses independently per `building_uid`: 216 candidate buildings, 595,973 building-coordinate hypotheses. Paper thresholds DA<=0.4 and mean coherence>=0.75 retain 25,358 pixels in 198 buildings.
- Every pair/building was unwrapped with actual GAMMA `mcf`, `tri_mode=1` Delaunay. Consolidated audit contains 48 pairs and zero failed buildings. SBAS is a coherence-weighted two-parameter velocity/height solve with Bisquare 4.685 and at least 12 pairs.
- Final map height is the robust pixel-height span P95-P05. The CSV also preserves the paper-text 1.5xIQR median and original-MATLAB max-min range because they are not equivalent on the expanded mixed ground/facade/roof search strip. This correction is disclosed and is not presented as the paper's literal formula.
- Formal results: `zjc/strict_reproduction/results/paper_strict/building_height_final.gpkg` (126,626 features; 198 solved, 126,428 null), matching CSV, pixel NPZ, and six standalone Chinese SVGs under `results/figures/`. Of the 198 numerical solutions, 128 pass the frozen internal quality gate.
- `Floor` is used only to recover the paper/original-code high-rise candidate and search-mask geometry. It is absent from MCF/SBAS and final aggregation. `Floor*3m` is introduced only after results freeze in `floor_prior_posthoc_audit.csv`; it never fills or corrects a result.
- Final structural QA: `zjc/strict_reproduction/inventory/final_reproduction_validation.json`, `all_pass=true`.
- Read first: `zjc/strict_reproduction/docs/01_严格复现执行结果与审计.md`. It records the high residuals, poor post-hoc agreement to Floor, missing LiDAR/79-building validation list, and the boundary between executable reproduction and claims that cannot be independently reproduced.

## Latest optimization — guarded dual-network upper-roof plateau

- The paper's `1.5 x IQR + median` remains the default building aggregation. A new guarded extension replaces it only when the 48-pair and 36-pair GAMMA solutions both contain the same spatially connected upper roof plateau. Frozen gates: spatial link <=3.1 px, height link <=3.0 m, >=4 points, >=20% support, plateau IQR <=3 m, uplift 1.5-12 m, cross-network plateau difference <=3 m, and point Jaccard >=0.50. No vector/prior height is read before selection and height freezing.
- 16/230 stable buildings pass the upper-plateau gate: clean IDs 145, 157, 183, 194, 199, 211, 492, 562, 568, 749, 750, 760, 773, 797, 856, 883. Median/P95/max uplift = 2.81/7.99/8.32 m. All others retain the paper median.
- New formal product: `results/geodata/tongji_building_height_guarded_upper_roof_consensus_gamma100.geojson`, `results/tables/tongji_building_height_guarded_upper_roof_consensus_gamma100.csv`, and `results/metadata/tongji_building_height_guarded_upper_roof_consensus_gamma100_summary.json`.
- New statistics: 230 solved / 798 unsolved; median/P05/P95/max = 17.21/5.99/47.66/86.54 m; no prior filling. Median roof IQR improves from 2.77 to 2.58 m. Accepted-product network absolute-difference median changes from 0.72 to 0.79 m while P95 remains 2.25 m; median phase sigma changes from 0.200 to 0.204 rad. The method targets mixed/multi-level roof aggregation, not shared integer-cycle aliases.
- An 8-sector equal-weight local-ground branch was implemented and rejected. Its cross-network absolute difference worsened slightly and 73/75 accepted buildings in the two networks had >2 m leave-one-sector variation. The original local point-median ground reference remains formal.
- Updated standalone SVGs: Figures 01, 12, 14 and 17. Figures 10 and 11 remain pre-inversion island evidence and were not changed. XML parsing and CairoSVG raster QA passed.
- Detailed record: `docs/双网络空间连续上层屋顶平台优化_20260824.md`.

## Latest optimization — dual-network GAMMA-SBAS stability consensus

- The local-roof-coherence `mb_pt` candidate is formally rejected. Although median fitted phase sigma fell from 0.200 to 0.178 rad, the accepted-building median rose from 16.53 to 23.66 m, P95 rose from 47.69 to 69.26 m, median roof IQR rose from 2.70 to 3.93 m, and an approximately 150 m candidate appeared. Roof-derived layer weights were applied to both roof and ground, breaking common-mode cancellation in the roof-minus-ground height.
- GAMMA documentation confirms that `mb_pt sigma` is one phase standard deviation per input interferogram layer and recommends simulated-minus-observed residuals for iterative improvement. A stable-ground residual reweighting pilot was implemented but rejected: 39/50 accepted, P95 54.07 m, median phase sigma 0.199 rad.
- A result-independent 36-pair perturbation network was built by dropping lowest-quality pairs only while the 22-date graph retained edge connectivity >=2, no bridges, and minimum degree >=2. It uses no building height. Builder: `code/reproduce_tongji/build_ipta_pair_stability_subset.py`; audit: `results/metadata/tongji_paperquality_stability36_network_summary.json`.
- Full equal-weight 36-pair GAMMA rerun: 237 accepted; median/P05/P95 = 16.61/5.56/47.41 m. Against the 48-pair solution there are 232 common solutions, median signed difference -0.38 m and P95 absolute difference 2.36 m.
- The formal consensus freezes a 4.0 m gate from the 50-building pilot (`|median difference| + 3 x robust MAD = 3.83 m`, rounded before the full run), retains 230 common stable buildings, and uses each building's median of the 48- and 36-pair equal-weight GAMMA solutions. No prior height is used for this gate or aggregation; no unsolved building is filled. The product remains hybrid because R-D roof positioning and integer-cycle initialization use vector height.
- Current formal product: `results/geodata/tongji_building_height_network_consensus_gamma100.geojson`, `results/tables/tongji_building_height_network_consensus_gamma100.csv`, and `results/metadata/tongji_building_height_network_consensus_gamma100_summary.json`.
- Current statistics: 230 solved / 798 unsolved; median/P05/P95 = 16.67/5.93/47.66 m; maximum = 86.54 m; `filled_from_prior=0` for all 1,028 buildings.
- Updated SVGs: Figures 01, 03, 03a, 12, 14, and 17. Figure 03 is the 48-pair main inversion network; the new standalone Figure 03a is the 36-pair stability network. Figure 17a is visibly marked as rejected. Figures 10 and 11 remain pre-inversion island-selection evidence and are deliberately not filtered by final height availability.

## Latest optimization — paper-weighted island network and coherence P matrix

- Implemented a paper-aligned island unwrap in `code/reproduce_tongji/benchmark_paper_unwrap.py`: only finite in-mask pixels are graph nodes; edge weights combine interferometric coherence, amplitude dispersion, and Bisquare residual weight; each connected component retains an observed wrapped-phase datum instead of an arbitrary zero.
- Corrected the building common-height search to average unit phase vectors weighted by coherence and DA. The previous code counted interferogram magnitude in addition to coherence. A frozen 36-building pure-InSAR rerun still collapsed to a 4.16 m median; the six low-sensitivity coarse pairs had weak distant-solution margins. This branch remains rejected.
- The weighted dense-island rerun for building 45 retained only three final paper-quality points, so it is rejected; it did not restore the high-rise top.
- Added local-roof-coherence phase sigmas for the GAMMA `mb_pt` P matrix, plus optional tempering and ratio caps. Full local weighting accepts 219 buildings and reduces median phase sigma from 0.200 to 0.178 rad, but increases median roof IQR from 2.61 to 3.97 m and produces a 150 m candidate. Tempering and ratio caps do not dominate it. None replaces current Figure 17.
- Rejected diagnostic map: `picall/17a_相干性加权屋顶核心区SBAS建筑高度分布.svg`; 219 solved, 809 unsolved, no prior filling. It is retained only as a failed ablation record.
- Full record: `docs/论文加权孤岛解缠与相干性权矩阵优化_20260824.md`.
- GAMMA runners now recreate `/tmp/gamma_gdal_compat/libgdal.so.26` automatically after temporary-directory cleanup.

The workspace is on the literature-aligned GAMMA/InSAR building-height route. The dual-network consensus above is the current formal hybrid product; older 238-building and 219-building products are retained only as superseded/rejected diagnostics.

Main documentation:

- `docs/zjc_method_code_deep_study_and_tongji_reproduction.md`（`zjc` 两份文献、85 个 MATLAB 文件、版本差异与同济严格复现设计）
- `docs/current_literature_route_and_optimization_plan.md`
- `code/reproduce_tongji/README.md`

## Latest checkpoint — read this first

- Figure 17 has now been updated from the adaptive-window roof-projection, 48-pair, paper-strict hybrid GAMMA-SBAS product (361 solved, 667 unsolved); do not present it as an independently validated pure-InSAR accuracy product.
- Do **not** copy the vector `height` field into a result, use it to fill an unsolved building, or use comparison error against it for QC/parameter selection.
- The user-authorized hybrid branch may use vector height to construct the initial 3-D GAMMA R-D roof projection and initialize integer phase cycles. It must be labelled hybrid/non-independent; final numerical height is re-estimated by GAMMA `mb_pt` and unsolved buildings remain `NaN`.
- Current strongest paper-strict hybrid candidate: 361 accepted buildings, median/P05/P95 = 16.05/5.15/47.74 m, median phase residual = 0.191 rad. It uses the adjusted roof-only projection, 48 pairs, point mean coherence >=0.75, at least 4 roof PS, phase sigma <=0.75 rad, roof IQR <=8 m, and `mb_pt gamma=100`.
- Broader hybrid first pass: 482 accepted, median/P05/P95 = 18.43/6.36/42.59 m, median phase residual = 0.325 rad. Retained for diagnostic coverage only.
- Damped R-D iteration (`alpha=0.6`) was rejected: 20 first-pass solutions were lost; only 79/462 common buildings met both <=1 m height change and <=0.5 pixel roof-centroid shift (17.1%). Retain the first-pass strict geometry.
- Earlier 37-pair `gamma=10` versus `gamma=100` testing retained 142 versus 319 buildings and selected `gamma=100` on frozen internal criteria; the current adaptive-window 48-pair `gamma=100` rerun supersedes those counts.
- The closure-corrected variant is more network-consistent but not more accurate in height scale: 299 accepted, median/P05/P95 = 13.80/1.86/30.17 m, median residual = 0.291 rad.
- A lower phase residual is not sufficient evidence of correct height. Several rejected branches converge cleanly to a wrong low-height ambiguity.
- Building 45 is recovered as 52.74 m by the broad prior-roof cycle initialization followed by GAMMA `mb_pt`; its median phase residual is 0.343 rad. Its roof-point median coherence is only 0.737, so it is excluded from the paper-strict mean-coherence >=0.75 product. The 52.74 m value is a hybrid candidate, not independent confirmation of the 51 m initializer.
- Required independent constraint for a defensible high-rise accuracy claim: more high-quality long-baseline pairs, another look direction/track, or LiDAR/survey control. Existing priors may initialize the authorized hybrid branch, but cannot also serve as its independent validation.

Latest detailed record:

- `docs/gamma_native_sbas_reconstruction_20260823.md`, especially sections 9–10.

Latest diagnostic artifacts:

- Closure-corrected result: `results/tables/tongji_building_height_gamma_roofcore_closurecorrected_gamma100_insar_only.csv`
- Closure summary: `results/metadata/gamma_punw_closure_correction_summary.json`
- Height sensitivity summary: `results/metadata/gamma_phase_height_sensitivity_summary.json`
- Dense island / GAMMA comparison for building 45: `results/metadata/zjc_dense_island_gamma_sbas_45_directcompare_summary.json`
- Strict paper-network test for building 45: `results/metadata/zjc_dense_island_gamma_sbas_45_paper_network_summary.json`
- Rejected stable-top-point summary: `results/metadata/tongji_building_height_stable_rooftop_ps_summary.json`
- Current strict hybrid CSV: `results/tables/tongji_building_height_adaptive_window_network48_gamma100_strict.csv`
- Current strict hybrid full vector: `results/geodata/tongji_building_height_adaptive_window_network48_gamma100_strict.geojson`
- Current strict hybrid summary: `results/metadata/tongji_building_height_adaptive_window_network48_gamma100_strict_summary.json`
- Rejected iteration audit: `results/metadata/tongji_prior_roof_gamma100_iteration_convergence_summary.json`

Current active development branch:

- `gamma_native_ipta_sbas` (experimental; not yet accepted as the final height product)
- GAMMA differential interferometry: `SLC_intf -> phase_sim_orb -> sub_phase -> adf`
- GAMMA IPTA unwrapping/initial model: `multi_def_pt`
- GAMMA SBAS time series and height correction: `mb_pt`
- local stable-ground subtraction from the same GAMMA solution
- IQR cleaning and building median/upper-cluster aggregation
- pure-InSAR branches merge vector `height` only after freezing the solution
- the explicitly labelled hybrid branch uses `height` only for initial roof geometry and integer-cycle initialization, then GAMMA `mb_pt` re-estimates height; it never fills missing values

Previous Python-heavy product (retained only for comparison, no longer active):

- GeoJSON: `results/geodata/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_insar_only.geojson`
- CSV: `results/tables/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_insar_only.csv`
- Summary: `results/metadata/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_summary.json`

Previous Python-heavy product statistics (comparison only; not current GAMMA result):

- Buildings total: 1028
- Strict InSAR solved: 260
- No strict solution: 768
- Median building height: 23.56 m
- P05/P95 building height: 4.48 / 49.28 m
- Negative / over-120 m accepted heights: 0 / 0

## GAMMA-native SBAS reconstruction (2026-08-23)

- Main code: `code/reproduce_tongji/run_gamma_native_ipta_sbas.py`
- Closure correction: `code/reproduce_tongji/correct_gamma_punw_closure.py`
- GAMMA height sensitivity: `code/reproduce_tongji/simulate_gamma_height_sensitivity.py`
- Dense ZJC-island/GAMMA diagnostic: `code/reproduce_tongji/run_zjc_dense_island_gamma_sbas.py`
- Stable rooftop-PS diagnostic: `code/reproduce_tongji/select_stable_roof_top_points.py` (rejected as final)
- Method log: `docs/gamma_native_sbas_reconstruction_20260823.md`
- The former `20200708_dsm_rdc.hgt` route is rejected because the raster contains building prior heights (P95 25 m, max 106 m). It must not be used as reference height plus residual.
- A spatially constant 4 m raster is used only as a non-NULL GAMMA phase datum. Final height is roof/support-point elevation minus local stable-ground elevation, so the constant cancels.
- `SLC_diff_intf` is incompatible with these TSX `SCOMPLEX` crops and writes all zeros in the installed GAMMA 2021 build. The documented equivalent `SLC_intf -> sub_phase` is used.
- True GAMMA SBAS requires `mb_pt`; `multi_def_pt` alone is only the IPTA initial unwrapping/model step.
- Previous global diagnostic baseline: 37 pairs, 22 dates, layover support, `mb_pt gamma=10`, 628 numerical building estimates, median phase residual 0.70 rad, height median 11.30 m. It has been superseded by the roof-core per-building `mb_pt(gamma=100)` experiments below and was never accepted as final.
- The 71-pair network is split into disconnected time components. The 2020-2021 component is rejected for ambiguity failure (P95 213 m). The 2022-2023 component is stable but still height-biased low.
- The proposed direct per-building MCF step has now been tested and rejected: it suppresses the roof-ground phase jump (building 658: 8.88 m versus 21.26 m when the global baseline-time ambiguity is retained). Direct local `multi_def_pt` and ZJC graph-unwrapping-to-`mb_pt` also collapse absolute height despite low residuals.
- Current best internally stable roof-core engineering candidate: global GAMMA `multi_def_pt` ambiguity initialization, per-building GAMMA `mb_pt(gamma=100)`, local stable-ground subtraction, upper roof cluster plus IQR. It accepts 318 buildings; median/P05/P95 = 14.13/1.88/35.94 m and median phase residual = 0.268 rad. It is still diagnostic, not final.
- Paper-strict roof-core candidate (`mean coherence >= 0.75`, IQR then median): 231 accepted buildings; median/P05/P95 = 8.83/0.86/30.86 m; median phase residual = 0.252 rad. The height scale remains too low.
- Current next issue is no longer the GAMMA SBAS call itself; it is absolute integer ambiguity initialization for roof-versus-ground height. Do not select a lower-residual branch if it collapses height scale.
- Network closure correction reduces triangle integer failures to about 0.11%, but the full corrected branch still has only 299 accepted buildings and compressed high-rise scale (median residual 0.291 rad); closure is not the main bottleneck.
- The paper text and original MATLAB differ: the paper aggregates full-island pixel elevations with IQR+median, while original code uses `max-min`; `unwrap_phase_matrix.m` also includes zeroed pixels outside the island inside its bounding rectangle.
- Dense full-island ZJC unwrap followed by GAMMA `mb_pt` does not recover building 45 (robust span about 7.7 m). The ZJC direct two-parameter solve agrees, so GAMMA is not suppressing the height.
- A 50.2 m candidate PS exists for building 45 and is stable across time/baseline subsets, but it is a singleton and cannot be distinguished from a stable integer alias. The full highest-stable-PS branch (332 buildings, P95 82.3 m) failed frozen post-comparison and is rejected.
- The former 2026-08-23 48-pair map contained 389 solved and 639 unfilled buildings. It is superseded by the 2026-08-24 adaptive-window full rerun (361/667); only the explicitly hybrid/non-independent GAMMA branch may feed the current height map, and rejected pure-InSAR diagnostics must not replace it.
- Frozen 36-building ambiguity perturbations prove that the hybrid solution remains initializer-cycle locked: `H0+50 m` shifts accepted results by a median `+50.51 m`; `H0-50 m` leaves only 2 accepted and shifts them by `-52.25 m`. Therefore the map is a prior-assisted residual estimate, not independent InSAR validation.
- The 48-pair QC network has 22 dates, 48 edges, edge connectivity 2, no bridges, and minimum degree 2. It was obtained from a 60-pair candidate network by dropping the 12 lowest-coherence edges subject to graph constraints; no height was used in network selection.
- Rejected new pure-InSAR diagnostics: height-position joint search over a common 0-180 m GAMMA R-D envelope (36 frozen buildings, 18 split-validated accepted, median 4.57 m), and `multi_def_pt` models 3/4 without a constant (11 accepted, median about 2.34 m). Low residual and pair-split stability do not remove the ground/low-height alias.

Important rule:

- Pure-InSAR diagnostics must not use vector `height` for fitting, ambiguity selection, filtering, filling, or QC.
- The user-authorized hybrid route may use it for initial building-body projection and integer-cycle initialization only. Such output must carry `insar_only=false`, `filled_from_prior=false`, and a non-independent comparison warning.
- Never output the initializer for an unsolved building. Never choose parameters using error against that same initializer.

## ZJC original-code reproduction (2026-08-22)

- Code: `code/reproduce_zjc_tongji/`
- Final figures: `picall/` (18 standalone SVG files only; filenames and visible labels are Chinese)
- GAMMA physical projection cache: `work/zjc_original_reproduction/20200708_all_building_gamma_corrected_projection_sar.geojson` (1028 valid ground/roof/support triplets; GAMMA `coord_to_sarpix` + `DIFF_par`).
- Figure 18 now uses the roof-only product `work/zjc_original_reproduction/20200708_all_building_adaptive_window_roof_projection_sar.geojson`; ground and layover surfaces are not drawn or exported. It starts from the prior GAMMA R-D + frozen-temporal roof projection and searches a per-building translation window whose half-width is `ceil(1.5 + 0.045*roof_max_dimension_px + 0.035*height_m)`, clipped to 3-9 pixels. Vector height is used only to size the search window, never as a match score, result filter, estimated height, or fill. The 20200708 full-precision RSLC roof boundary/local contrast is the primary score; 21 frozen dates excluding 20200708 provide the acceptance gate. 125/682 visible roofs pass; median primary/validation gains are 0.0363/0.0444 and median shift is 3 pixels. Code: `code/reproduce_tongji/refine_projection_local_window_search.py`; metrics: `work/zjc_original_reproduction/20200708_all_building_local_window_projection_metrics.csv`; audit: `results/metadata/tongji_projection_local_window_search_summary.json`.
- Matched standalone roof-only comparison plates: `picall/18a_局部窗口搜索前建筑投影.svg` and `picall/18b_局部窗口搜索后建筑投影.svg`. Both use the identical 20200708 single-RSLC background transform, extent, size, colors, and line widths; the same 125 accepted roofs are highlighted in magenta. No ground or building-body support is shown. Generator: `code/reproduce_tongji/generate_projection_window_search_before_after_svgs.py`.
- Projection-refinement stages: `code/reproduce_tongji/refine_gamma_projection_with_sar_features.py` followed by `code/reproduce_tongji/refine_projection_local_window_search.py`; final Figure 18 audit: `results/metadata/tongji_projection_local_window_search_summary.json`.
- Figure 02 (`picall/02_同济雷达单景幅度.svg`) now uses only the original reference acquisition `20200708.rslc` (GAMMA `SCOMPLEX`, 630x900, 1-look). Its display reproduces the supplied GAMMA-style SAR reference: analysis of the reference BMP against its colocated full-precision RSLC gives gray level proportional to amplitude^0.70 for unsaturated pixels, followed by the reference plotting code's global 1%-99.4% stretch. The formal figure applies that transform directly to `20200708.rslc`; the BMP is not used as input. Temporal averaging, spatial filtering, sharpening, resampling, multilooking, interpolation, vector overlay, and local histogram equalization are prohibited. Code: `code/reproduce_tongji/build_rslc_mean_amplitude_svg.py`; array: `work/mli/20200708_rslc_amplitude.npy`; audit: `results/metadata/20200708_single_rslc_amplitude_summary.json`.
- Feature-refined projection evidence remains in Figure 18. Projection-dependent Figures 04-09 and 18 were regenerated as standalone Chinese SVGs; height-result Figures 10-17 were not silently recomputed from the changed masks.
- Numerical basis: ZJC `unwrap_phase_matrix.m` + `local_knn.m` + root `LGR_demerror_est.m`
- Original parameters retained: local 9-point neighborhoods, 4-neighbor KNN, Bisquare 4.685, unweighted LGR, `max-min` building-height range
- Tongji high-rise analogue subset: `Floor > 10`, 63 buildings / 67 islands. For Fig. 18, vector `height` only extrudes each building from 4 m ground height to `4 m + height`; it is never used for phase/LGR inversion, missing-height filling, projection-error fitting, selection, or QC.
- Input pairs: 37; ZJC solved buildings: 48
- ZJC height-range median/P95: 30.22 / 53.82 m
- Conventional full-scene unwrap height-range median/P95: 111.13 / 257.94 m
- Error-statistics figures 07–11 were removed at the user's request.

## Previous Python roof-core stable-ground branch (comparison only, 2026-08-23)

- Previous comparison product: `results/geodata/tongji_building_height_roof_core_stable_ground_temporal_multistart_weighted_median_insar_only.geojson`
- Building vector `height` is used only for the initial GAMMA roof geometry. It is not read by roof-core masking, stable-ground selection, SBAS inversion, ambiguity selection, physical QC, aggregation, or missing-height filling.
- Adaptive roof-core mask: 661 in-scene buildings, 687 clean-ID islands, 95,400 pixels; 1,532 conflicting pixels removed.
- Stable-ground reference: 7,359 pixels; each interferogram has 1,956–6,082 usable reference pixels.
- Solver: stable-ground circular phase anchoring, temporal multi-start ambiguity search, coherence/DA/pair/RMSE/baseline-span gates, physical-branch rejection, quality-weighted roof median.
- Strict solution: 260 solved / 768 unsolved; median/P05/P95 = 23.56/4.48/49.28 m; no negative or over-120 m accepted values.
- Damped projection iteration (`alpha=0.6`) was rejected: 38 initial solutions were lost and only 11/222 common solutions met both height and pixel-shift convergence criteria. This branch is retained only for comparison and is not the active GAMMA result.
- The 18 paper-reference figures in `picall/` were generated from this now-rejected Python-heavy branch and are retained only for comparison. They have not been updated with the experimental GAMMA-native result.
- The former 260-, 319-, and 389-solution Figure 17 products remain comparison history; the current adaptive-window product contains 361 strict hybrid GAMMA-SBAS solutions.

## Current updated figure set (2026-08-24)

Superseding result after re-reading the paper and correcting the implementation (2026-08-24):

- **Status correction (later on 2026-08-24): the 238-building map is not accepted as a prior-independent height product.** It remains an explicitly prior-assisted diagnostic because roof R-D position and integer-cycle initialization use the vector height. Do not present it as independent InSAR validation.
- Prior-independent audit: `docs/纯InSAR建筑高度先验独立性审计_20260824.md`.
- A true 71-pair IPTA stack was rebuilt after discovering that the old `gamma_native_ipta_sbas_full71/ipta/pairs.itab` contained only 17 records although its summary reported 71.
- Pure-InSAR branches tested and rejected include multi-temporal amplitude/R-D height search, true-71-pair global `multi_def_pt`, per-point wrapped search, split-baseline stability gating, flat-roof common-height search, 0-120 m height-position coupling, full-island `mcf_pt -> mb_pt`, and original-code-style island `max-min`.
- None of those branches may replace Figure 17. Their frozen post-solution correlations to the prior ranged from -0.78 to 0.17 (the best tested local-ground point branch reached 0.52 but MAE remained 27.93 m); the full-island `max-min` branch had median span 7.98 m and correlation 0.11.
- Additional code fixes: valid-ground points are now filtered before building the nearest-neighbour tree; the common-height branch supports local wrapped-zero ground initialization; the wrapped search supports common prior-independent intervals, fixed split-baseline validation, distant-alias cost margin, and sensitivity-sign audit.

- Active strict product: `results/geodata/tongji_building_height_paperquality_gamma100.geojson` and `results/tables/tongji_building_height_paperquality_gamma100.csv`.
- Active mask/projection: `work/roof_sbas_adaptive_window_paper_quality_network48/` and `work/zjc_original_reproduction/20200708_all_building_adaptive_window_paper_quality_gated_roof_projection_sar.geojson`.
- A bug that averaged only coherence values already above 0.55 was corrected. Mean coherence is now computed over every finite interferogram; the count above 0.55 remains a separate >=12-pair criterion.
- Local SAR-feature shifts are accepted only when the number of paper-quality roof pixels (DA<=0.40, all-pair mean coherence>=0.75, >=12 pairs with coherence>=0.55) does not decrease. Accepted local shifts changed from 125 to 88.
- Final strict GAMMA branch: roof-only one-pixel eroded/conflict-free cores, per-building local stable ground, geometry-cycle initialization, GAMMA `mb_pt(gamma=100)`, local-ground subtraction, 1.5xIQR cleaning, building median. No unsolved height is filled.
- Active statistics: 238 solved / 790 unsolved; median/P05/P95 = 16.53/5.83/47.69 m; median fitted phase sigma = 0.200 rad. The prior comparison is explicitly non-independent (MAE 3.65 m, correlation 0.955) and is not a parameter-selection target.
- Rejected tests: paper-threshold network weighting (only 15/48 edges compliant; phase sigma worsened to 0.213 rad and median collapsed to 7.62 m), local-roof coherence weighting (coverage and error worsened), smaller local-ground windows (higher phase residual), and `mb_pt gamma` below 100 (higher phase residual). GAMMA rejects gamma>100.
- Projection-height iteration was run, not assumed: 238 roofs moved by median 2.66 pixels using the first GAMMA solution. Iteration 2 fell to 211 solutions, lost 37, gained 10, and only 11.4% of common buildings converged within 1 m and 0.5 pixel. It is rejected; audit: `results/metadata/tongji_sbas_iteration1_convergence_summary.json`.
- Formal SVGs regenerated from the active 238-building product: Figures 01, 10, 11, 12, 14, and 17. Figure 17 labels every solved building with its GAMMA-estimated value; unsolved buildings remain grey/transparent and unfilled.

- SVG: `picall/17_屋顶核心区SBAS建筑高度分布.svg`
- Generator: `code/reproduce_tongji/generate_strict_gamma_height_map_svg.py`
- Related-figure generator: `code/reproduce_tongji/generate_strict_gamma_related_svgs.py`
- Source vector: `results/geodata/tongji_building_height_paperquality_gamma100.geojson`
- 1028 building geometries; 238 solved and labelled; 790 unsolved shown in grey and never filled.
- SVG QA: XML parsing and CairoSVG raster rendering succeeded; Chinese title/note, north arrow, 200 m scale bar, height values, and colorbar render correctly.
- Updated result-dependent SVGs: Figures 01, 10, 11, 12, 14, and 17. Figures 02–09, 13, 15, 16, and 18 retain their stated amplitude/baseline/overlap/global-unwrap/prior/projection roles.
- Figures 10 and 11 now use the adaptive-window roof-only projection and remain pre-inversion island-selection products, not plots filtered by final solved height. They retain 553 buildings, 575 islands, 87,446 island-domain pixels and 55,504 reliable evidence pixels; 1,829 overlapping-roof conflict pixels are removed. Final height availability is not a selection criterion, and vector height is not used as a matching target or fill. Builder: `code/reproduce_tongji/build_roof_core_reference_masks.py`; figure generator: `code/reproduce_tongji/generate_optimized_building_island_overview_svg.py`; cache: `work/roof_sbas_adaptive_window_network48/`; audit: `results/metadata/figure10_optimized_building_islands_summary.json`.
