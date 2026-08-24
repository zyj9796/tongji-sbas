# 同济区域 GAMMA-SBAS 建筑高度复现项目

本目录保留同济区域建筑高度反演的原始数据、文献、ZJC原始代码、完整研究代码、历史消融记录、当前正式结果及复现所需中间数据。可再生成的运行缓存和临时文件已清理。

## 当前正式结果

- 建筑高度表：`results/tables/tongji_building_height_guarded_upper_roof_consensus_gamma100.csv`
- 全量建筑矢量：`results/geodata/tongji_building_height_guarded_upper_roof_consensus_gamma100.geojson`
- 审计摘要：`results/metadata/tongji_building_height_guarded_upper_roof_consensus_gamma100_summary.json`
- 正式独立SVG：`picall/`
- 当前方法说明：`docs/双网络空间连续上层屋顶平台优化_20260824.md`
- 工作状态与约束：`agent.md`

## 目录说明

| 目录 | 内容 | 清理策略 |
|---|---|---|
| `data/` | 同济RSLC、DEM/DSM和建筑矢量 | 原始输入，保留 |
| `文献和原始代码/` | 论文与GAMMA中文手册等 | 方法依据，保留 |
| `zjc/` | ZJC原始MATLAB代码、论文和原实验数据 | 原始方法依据，保留 |
| `code/` | 同济复现、GAMMA处理和制图代码 | 保留；删除`__pycache__` |
| `work/` | GAMMA/IPTA中间结果、逐建筑点结果和历史试验 | 保留研究过程；仅删除空临时目录 |
| `results/` | 正式结果、历史结果、诊断与审计 | 保留 |
| `picall/` | 当前论文参考图，单图SVG输出 | 保留 |
| `docs/` | 文献学习、方法演变、失败分支和优化记录 | 保留 |
| `.venv/` | 当前Python复现环境 | 保留，便于直接复现 |
| `results/logs/` | 从项目根目录归档的GAMMA运行日志 | 保留 |

## 重要方法约束

- `clean_equal_height.height`可以用于已授权混合路线的屋顶R-D定位与整数周初始化，但不得用于结果填充、反演值替代、筛选目标或参数调优。
- 无解建筑保持`NaN`。
- 当前建筑聚合默认遵循论文的`1.5×IQR + 中位数`；只有两个冗余网络共同确认的空间连续上层屋顶平台才允许替换。
- `picall/`中的正式图片只输出SVG，图中文字和文件名使用中文。

