# Footprint-Constrained 论文 Fig. 4 复现说明

更新时间：2026-08-25

## 图件含义

论文 Fig. 4 的图注为 “Examples of building overlap mask”。该图不是建筑高度结果，而是建筑轮廓经雷达坐标投影后，对相邻建筑支持域重叠进行识别与剔除的示例。

复现图保持原图的两列三行结构：

1. 原始雷达幅度；
2. 建筑投影支持域及黑色斜线重叠区；
3. 从所有相关建筑中同时剔除重叠像元后得到的独立支持域。

## 实际数据

- SAR 背景：2023-10-07 单景幅度，局部窗口使用原始像元。
- 支持域：`independent_expanded_search_points.npz` 中按 `building_uid` 独立保存的雷达坐标投影假设。
- 案例一：建筑 UID 67652、67653、67654；三个支持域分别包含 3372、3291、3417 个像元，共识别 1873 个重叠像元。
- 案例二：建筑 UID 94855、97340；两个支持域分别包含 5757、5450 个像元，共识别 1464 个重叠像元。
- 案例只按雷达像元重叠关系选取，没有读取或比较最终建筑高度。

## 输出与质量检查

- 正式输出：`results/footprint_constrained_paper/图4_建筑投影重叠掩膜示例.svg`
- 生成器：`code/make_footprint_paper_fig4.py`
- 单一字体：`Noto Sans CJK SC`。
- 8 个中文标注均为可编辑 SVG 文本。
- SVG 通过 XML 解析和 1800 像素宽渲染检查；没有输出正式 PNG/PDF。

## 复现命令

```bash
MPLCONFIGDIR=/tmp/mpl-footprint .venv/bin/python \
  zjc/strict_reproduction/code/make_footprint_paper_fig4.py
```
