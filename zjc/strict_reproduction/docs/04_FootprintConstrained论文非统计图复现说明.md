# Footprint-Constrained 论文非统计图复现说明

更新时间：2026-08-25

## 输出范围

正式目录：`results/footprint_constrained_paper/`。

按论文原始编号输出 8 张独立中文 SVG：Fig. 1、2、3、4、5、6、7、13。遵照此前要求，未输出 Fig. 8—12 的精度散点、残差直方图、残差趋势、箱线图和误差 CDF。

## 图件与数据来源

- **图 1**：复现论文四模块 2×2 总流程。流程内嵌真实 SAR 幅度、天津建筑轮廓、雷达建筑支持域、代表性干涉对缠绕相位和当前有解建筑高度，不使用论文截图。
- **图 2**：使用 2023-10-07 实际 SAR 幅度和实验区覆盖框。论文使用的卫星光学底图未随数据提供，因此没有伪造或联网替换底图。
- **图 3**：直接读取 `paper_result_48.bperp`，绘制 21 个日期、48 条干涉边的时空基线网络。
- **图 4**：使用真实建筑投影支持域，展示重叠识别和双向剔除；详细审计见 `03_FootprintConstrained论文Fig4复现说明.md`。
- **图 5**：全景使用实际单景 SAR 幅度和建筑独立支持域，局部放大叠加论文阈值像元。
- **图 6**：左图为 `pixel_height_independent_mcf_fixed_far_48.npz`；右图为 `pixel_height_mcf_fixed_far_48.npz` 覆盖式 MCF 对照。流水线没有保留论文意义上的整幅全局 MCF 高程栅格，因此右图在图内明确标注“非整幅全局 MCF”，没有冒充论文的传统全局产品。
- **图 7**：宽幅三维城市白模。灰色低层背景采用统一 10 m 示意高度，橙色建筑采用本次 SBAS 有解值；统一示意高度只用于背景可视化，不填入任何数值结果。
- **图 13**：保持 CNBH-10 m、GBA、本文方法三列和上下两级尺度。CNBH/GBA 原始栅格未提供，前两列明确标注缺失；第三列使用真实 GAMMA-MCF+SBAS 建筑结果。

## 复现边界

本轮有三项输入缺失，不能声称像元级严格复现：Fig. 2 的卫星光学底图、Fig. 6 的整幅传统全局 MCF 高程栅格、Fig. 13 的 CNBH/GBA 高度产品。相关图均使用可审计替代或显式占位，不生成虚假数据。

## 格式与检查

- 输出数量：8 张 SVG；没有正式 PNG/PDF。
- 字体：`Noto Sans CJK SC`；SVG 文字保持可编辑。
- 密集 SAR、相位和点图层作为高清栅格嵌入 SVG；流程框、箭头、坐标、图例和文字保持矢量。
- 8/8 文件通过 XML 解析及最高 1800 像素宽 Python/CairoSVG 渲染检查。
- 图 1 长标题经过分行与内容区下移处理；图 6 增加双面板间距；图 7 使用无多余页边的宽幅三维布局。

## 复现命令

```bash
MPLCONFIGDIR=/tmp/mpl-footprint-all .venv/bin/python \
  zjc/strict_reproduction/code/make_footprint_paper_other_figures.py
```

Fig. 4 也可单独运行：

```bash
MPLCONFIGDIR=/tmp/mpl-footprint .venv/bin/python \
  zjc/strict_reproduction/code/make_footprint_paper_fig4.py
```
