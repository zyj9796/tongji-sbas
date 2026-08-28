#!/usr/bin/env node
"use strict";

const fs = require("fs");
const path = require("path");
const { execFileSync } = require("child_process");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel,
  ImageRun, LevelFormat, PageBreak, PageNumber, Packer, Paragraph,
  ShadingType, Table, TableCell, TableRow, TextRun, WidthType,
} = require("docx");

const root = path.resolve(__dirname, "..");
const outputDir = path.join(root, "results", "report");
const tempDir = "/tmp/tianjin_reproduction_docx_assets";
fs.mkdirSync(outputDir, { recursive: true });
fs.mkdirSync(tempDir, { recursive: true });

const outputPath = path.join(outputDir, "天津建筑三维重建论文复现总结报告_通俗版.docx");
const font = "Microsoft YaHei";
const colors = {
  navy: "1F4E79", blue: "2F75B5", teal: "2A9D8F", orange: "E67E22",
  paleBlue: "EAF2F8", paleTeal: "E8F5F2", paleOrange: "FDF0E6",
  paleGray: "F3F5F7", gray: "5B6573", dark: "1F2933", white: "FFFFFF",
};
const contentWidth = 9506;

function renderSvg(svgRelative, pngName, width) {
  const src = path.join(root, svgRelative);
  const dst = path.join(tempDir, pngName);
  if (process.env.DOCX_SKIP_RENDER === "1") {
    if (!fs.existsSync(dst)) throw new Error(`missing pre-rendered asset: ${dst}`);
    return dst;
  }
  const py = [
    "import cairosvg,sys",
    "cairosvg.svg2png(url=sys.argv[1], write_to=sys.argv[2], output_width=int(sys.argv[3]))",
  ].join(";");
  execFileSync(path.resolve(root, "..", "..", ".venv", "bin", "python"), ["-c", py, src, dst, String(width)]);
  return dst;
}

const images = {
  workflow: renderSvg("results/paper_figure_reproduction/图3.1_技术路线图.svg", "workflow.png", 1200),
  projection: renderSvg("results/figures/02_全部建筑矢量投影至雷达坐标.svg", "projection.png", 1600),
  phase: renderSvg("results/paper_figure_reproduction/图3.8_建筑区域独立相位解缠结果.svg", "phase.png", 1800),
  height: renderSvg("results/figures/05_建筑高度反演图.svg", "height.png", 1500),
};

function run(text, options = {}) {
  return new TextRun({ text, font, size: 22, color: colors.dark, ...options });
}

function body(text, options = {}) {
  return new Paragraph({
    alignment: options.alignment || AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: 150 },
    indent: options.noIndent ? undefined : { firstLine: 440 },
    keepNext: options.keepNext || false,
    children: [run(text, options.run || {})],
  });
}

function heading(text, level = 1) {
  return new Paragraph({
    heading: level === 1 ? HeadingLevel.HEADING_1 : HeadingLevel.HEADING_2,
    children: [new TextRun({ text, font, bold: true, color: level === 1 ? colors.navy : colors.blue })],
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { line: 330, after: 100 },
    children: [run(text)],
  });
}

function numbered(text) {
  return new Paragraph({
    numbering: { reference: "steps", level: 0 },
    spacing: { line: 340, after: 130 },
    children: [run(text)],
  });
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "D3DBE3" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, width, options = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    borders,
    shading: options.fill ? { fill: options.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 110, bottom: 110, left: 140, right: 140 },
    verticalAlign: "center",
    children: [new Paragraph({
      alignment: options.alignment || AlignmentType.LEFT,
      spacing: { after: 0 },
      children: [run(text, { bold: options.bold || false, color: options.color || colors.dark, size: options.size || 20 })],
    })],
  });
}

function dataTable(headers, rows, widths) {
  return new Table({
    width: { size: contentWidth, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, widths[i], { fill: colors.navy, bold: true, color: colors.white, alignment: AlignmentType.CENTER })) }),
      ...rows.map((row, ri) => new TableRow({ children: row.map((value, i) => cell(String(value), widths[i], { fill: ri % 2 === 0 ? colors.white : colors.paleGray, alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) })),
    ],
  });
}

function callout(title, text, fill = colors.paleBlue, accent = colors.blue) {
  return new Table({
    width: { size: contentWidth, type: WidthType.DXA },
    columnWidths: [contentWidth],
    rows: [new TableRow({ children: [new TableCell({
      width: { size: contentWidth, type: WidthType.DXA },
      borders: {
        top: { style: BorderStyle.SINGLE, size: 3, color: accent },
        bottom: { style: BorderStyle.SINGLE, size: 3, color: accent },
        left: { style: BorderStyle.SINGLE, size: 14, color: accent },
        right: { style: BorderStyle.SINGLE, size: 3, color: accent },
      },
      shading: { fill, type: ShadingType.CLEAR },
      margins: { top: 150, bottom: 150, left: 180, right: 180 },
      children: [
        new Paragraph({ spacing: { after: 80 }, children: [run(title, { bold: true, color: accent, size: 23 })] }),
        new Paragraph({ spacing: { line: 330, after: 0 }, children: [run(text, { size: 21 })] }),
      ],
    })] })],
  });
}

function figure(imagePath, width, height, caption) {
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 80 },
      children: [new ImageRun({
        type: "png", data: fs.readFileSync(imagePath),
        transformation: { width, height },
        altText: { title: caption, description: caption, name: caption },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 180 },
      children: [run(caption, { italic: true, color: colors.gray, size: 19 })],
    }),
  ];
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

const children = [];

// Cover
children.push(
  new Paragraph({ spacing: { before: 1500, after: 280 }, alignment: AlignmentType.CENTER, children: [run("天津建筑三维重建", { size: 46, bold: true, color: colors.navy })] }),
  new Paragraph({ spacing: { after: 520 }, alignment: AlignmentType.CENTER, children: [run("论文复现总结报告（通俗版）", { size: 34, bold: true, color: colors.blue })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, border: { bottom: { style: BorderStyle.SINGLE, size: 18, color: colors.teal, space: 1 } }, spacing: { after: 650 }, children: [run(" ", { size: 4 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 180 }, children: [run("主要内容：方法怎么做、结果怎么样、结果该怎么理解", { size: 24, color: colors.gray })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1100, after: 100 }, children: [run("复现目录：zjc/strict_reproduction", { size: 21, color: colors.gray })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [run("2026年8月", { size: 21, color: colors.gray })] }),
  pageBreak(),
);

children.push(
  heading("一、先说结论", 1),
  callout("一句话结论", "本项目已经从原始雷达影像出发，完整跑通了建筑轮廓投影、逐建筑相位解缠、SBAS 高程反演和建筑高度制图流程；但由于缺少 LiDAR 或逐栋实测高度，只能证明流程可执行，不能证明已经达到论文报告的外部精度。", colors.paleTeal, colors.teal),
  body("这次复现不是在已有高度图上重新配色，而是从 21 景复数 SAR 影像开始重新处理。所有没有得到可靠解的建筑都保留为空值，没有用楼层数、邻近建筑高度或外部产品补齐。"),
  heading("核心结果", 2),
  dataTable(
    ["项目", "结果", "怎么理解"],
    [
      ["SAR 影像", "21 景", "2023-10-07 至 2024-06-05"],
      ["正式干涉网络", "48 对", "用于主结果的时间序列相位观测"],
      ["候选高层建筑", "216 栋", "按论文和原代码规则恢复的研究对象"],
      ["得到数值解", "198 栋", "存在可计算的建筑高度结果"],
      ["通过内部质量门", "128 栋", "网络和模型内部较稳定，不等于外部精度合格"],
      ["全量建筑矢量", "126626 栋", "其中 126428 栋无解并保持空值"],
    ],
    [2900, 1900, 4706],
  ),
  body("198 栋有解建筑的高度中位数为 36.69 m，5% 和 95% 分位数分别为 4.36 m 和 95.00 m。这个分布描述的是本次反演结果，不应直接当作真实城市建筑高度分布。", { noIndent: true }),
  callout("最重要的解释", "“有解”表示计算流程得到了数值；“通过内部质量门”表示不同网络和模型指标较稳定；只有与独立 LiDAR 或实测高度对比后，才能判断真实精度。", colors.paleOrange, colors.orange),
  pageBreak(),
);

children.push(
  heading("二、方法到底做了什么", 1),
  body("可以把整套方法理解成五步：先把多期雷达影像对齐，再从不同日期的相位差中提取高程信息；同时把建筑轮廓投到雷达图像上；随后每栋建筑单独解缠；最后用 SBAS 把多期相位转换成高度。"),
  numbered("准备影像：每个日期有两段条带，先拼接，再裁剪出与论文一致的 10000×7000 像元研究区。"),
  numbered("配准和干涉：把 20 景辅影像精确对齐到 2023-10-07 参考景，并生成 48 对差分干涉图。"),
  numbered("定位建筑：把地理坐标中的建筑轮廓通过 GAMMA 几何查找表投影到雷达距离—方位坐标。"),
  numbered("逐栋解缠：每栋建筑单独建立相位网络，用 GAMMA MCF 解开相位的 2π 周期。"),
  numbered("SBAS 反演：综合 48 对相位、垂直基线和相干性，联合估计形变速度与高程残差，再汇总为建筑高度。"),
  ...figure(images.workflow, 360, 490, "图1  本次复现采用的总体技术路线"),
  pageBreak(),
);

children.push(
  heading("三、关键步骤的通俗解释", 1),
  heading("1. 为什么要精确配准", 2),
  body("SAR 相位对像元位置非常敏感。如果不同日期的同一建筑没有落在同一个像元上，后面的相位差就会混入位置误差。复现中 20 景辅影像的距离向配准标准差约为 0.0064—0.0124 像元，方位向约为 0.0073—0.0151 像元，明显优于预先设定的 0.1 像元门槛。"),
  heading("2. 建筑轮廓为什么不能直接盖在 SAR 上", 2),
  body("地图上的建筑轮廓是地面坐标，而 SAR 是侧视成像的距离—方位坐标。两者不是简单平移关系。本项目使用 SRTM、轨道参数和 GAMMA 查找表完成坐标转换，并把每栋建筑的支持域分别保存。这样即使相邻建筑在 SAR 中发生叠掩，也不会因为写入顺序让后一栋建筑抢走前一栋的像元。"),
  ...figure(images.projection, 610, 439, "图2  全部建筑轮廓投影至雷达坐标的结果"),
  heading("3. 什么是逐建筑相位解缠", 2),
  body("雷达只能直接观测到 −π 到 π 范围内的相位，真实相位可能多绕了若干圈。传统整幅图解缠容易让道路、地面和相邻建筑之间的错误传播到目标建筑。本方法把每栋建筑当成独立区域，使用 Delaunay 三角网络和 GAMMA MCF 单独求解整数周，从空间上限制错误传播。"),
  pageBreak(),
);

children.push(
  heading("四、怎样从相位得到建筑高度", 1),
  body("对一个建筑像元来说，不同干涉对的相位变化由两部分组成：一部分与时间有关，可表示平均形变速度；另一部分与垂直基线有关，可表示高程残差。SBAS 把 48 对观测放进同一个方程中，同时求出这两个未知量。相干性高的观测权重大，相干性低或残差异常的观测会被降低权重。"),
  callout("SBAS 模型", "相位 = 时间项 × 平均形变速度 + 基线敏感度 × 高程残差 + 噪声。复现采用相干性加权求解，并使用 Bisquare 权抑制异常残差。", colors.paleBlue, colors.blue),
  heading("像元如何筛选", 2),
  bullet("振幅离差 DA 不大于 0.4：多期亮度相对稳定。"),
  bullet("平均相干性不低于 0.75：相位在时间序列中较可靠。"),
  bullet("每个像元至少具有 12 对有效干涉观测。"),
  body("595973 个建筑—像元候选经过筛选后保留 25358 个高质量像元，覆盖 198 栋建筑。48 对、216 栋建筑的 GAMMA-MCF 任务全部执行完成，机器汇总中的失败数为 0。", { noIndent: true }),
  ...figure(images.phase, 620, 399, "图3  SAR 幅度、缠绕相位与两种 MCF 解缠结果对比"),
  pageBreak(),
);

children.push(
  heading("五、建筑高度为什么不用简单平均", 1),
  body("实际建筑搜索带中可能同时出现地面、立面、屋顶、叠掩和背景像元，因此这些像元不一定属于同一个等高面。论文正文建议先按 1.5 倍 IQR 去除异常值，再取中位数；原 MATLAB 则使用最大值减最小值。两者对应的物理含义并不相同。"),
  dataTable(
    ["汇总方法", "优点", "主要问题"],
    [
      ["IQR 后中位数", "不容易受少量异常值影响", "混合搜索带中可能只代表主模态，不一定是地面到屋顶高度"],
      ["最大值减最小值", "保留垂直跨度的直观含义", "一个错误周或噪声点就可能把高度拉得很大"],
      ["P95−P05", "保留稳健垂直跨度，降低极端值影响", "属于本项目披露的工程修正，不是论文原公式"],
    ],
    [2200, 3100, 4206],
  ),
  body("正式地图使用 P95−P05 作为推荐高度，同时在结果表中保留论文中位数和原代码全极差，方便后续比较。这样做没有读取楼层高度，也没有把先验值写回结果。"),
  callout("关于地面高度", "SRTM 地面椭球高程的中位数约为 −2.27 m。天津地区出现负椭球高是高程基准转换后的正常几何结果，不表示建筑位于地下。屋顶椭球高程等于地面椭球高程加反演建筑高度。", colors.paleTeal, colors.teal),
  pageBreak(),
);

children.push(
  heading("六、最终结果", 1),
  ...figure(images.height, 570, 431, "图4  本次 GAMMA-MCF + SBAS 建筑高度反演图"),
  dataTable(
    ["统计量", "建筑高度"],
    [
      ["5% 分位数", "4.36 m"],
      ["25% 分位数", "14.61 m"],
      ["中位数", "36.69 m"],
      ["75% 分位数", "59.64 m"],
      ["95% 分位数", "95.00 m"],
    ],
    [5000, 4506],
  ),
  heading("内部稳定性", 2),
  body("为检查论文中“48 对”和“时间基线不超过 44 天”的矛盾，项目另建了 45 对网络。48 对结果减去 45 对结果的建筑高度差中位数为 0.10 m，5% 和 95% 分位数为 −3.26 m 和 3.02 m。这说明额外的 3 个 55 天干涉对不是当前误差的主要来源。"),
  heading("相位残差", 2),
  body("25358 个像元的形式高度标准差中位数为 5.34 m，相位残差 RMS 中位数为 2.79 rad。残差仍然偏大，因此最终结果带有质量状态，不能只凭图面颜色判断高度是否准确。"),
);

children.push(
  heading("七、先验高度有没有影响结果", 1),
  callout("结论", "没有用 Floor×层高填充、校正或筛选正式高度。126428 栋无解建筑全部保持为空值。", colors.paleTeal, colors.teal),
  body("Floor 只用于恢复论文关注的高层候选建筑和原代码搜索带几何。SBAS 反演脚本不读取 Floor，建筑高度冻结后才单独生成楼层对照表。"),
  body("后验检查中，Floor×3 m 与正式结果的相关系数约为 0.44，中位绝对差约为 68.88 m。这个差异没有反向用于调参或修图。它提醒我们：当前结果与楼层信息并不一致，但楼层本身也不是独立测量真值，不能用它把结果改成“看起来合理”。"),
  heading("论文和原代码中最关键的差异", 2),
  dataTable(
    ["问题", "原文或原代码", "本次处理"],
    [
      ["干涉网络", "48 对与最大 44 天互相矛盾", "48 对主结果，45 对敏感性检查"],
      ["建筑重叠", "原代码按顺序覆盖", "每栋建筑独立保留支持域"],
      ["相位解缠", "正文 MCF，MATLAB 实际 LSMR", "正式调用 GAMMA mcf"],
      ["相位参考", "第一像元置零", "固定远距侧高质量参考点"],
      ["高度汇总", "正文中位数，代码全极差", "两者保留，正式图采用披露的 P95−P05"],
    ],
    [2100, 3300, 4106],
  ),
  pageBreak(),
);

children.push(
  heading("八、哪些结论现在还不能说", 1),
  body("当前资料缺少逐栋 LiDAR 或实测建筑高度、作者人工筛选的 79 栋验证建筑清单、论文版本的 GBA/CNBH-10 m 产品和整幅传统全景 MCF 高程栅格。因此以下说法目前没有足够证据："),
  bullet("不能说本次结果已经达到论文给出的高度精度。"),
  bullet("不能把 128 栋内部质量通过建筑称为“实测验证通过”。"),
  bullet("不能用 Floor×3 m 同时作为初始化、调参依据和验证真值。"),
  bullet("不能把缺失的 GBA、CNBH 或传统全景 MCF 面板补成看似完整的对比结果。"),
  heading("下一步最有价值的工作", 2),
  numbered("获得逐栋 LiDAR 或测量高度，并建立可靠的建筑 ID 对照。"),
  numbered("增加另一轨向或视向 SAR，帮助判断独立建筑的整数周歧义。"),
  numbered("增加高质量长垂直基线干涉对，提高高程敏感度。"),
  numbered("将新增真值分为调参与独立验证两部分，避免同一数据既参与估计又参与评价。"),
  callout("最终判断", "本项目完成的是“可执行、可追溯的论文方法复现”，而不是“论文外部精度的独立确认”。它的主要价值是跑通真实数据流程、找出论文与代码差异，并把不能验证的部分诚实保留下来。", colors.paleOrange, colors.orange),
  pageBreak(),
);

children.push(
  heading("九、主要成果文件", 1),
  dataTable(
    ["成果", "位置"],
    [
      ["像元级 SBAS 解", "results/paper_strict/pixel_height_independent_mcf_fixed_far_48.npz"],
      ["198 栋有解建筑表", "results/paper_strict/building_height_final.csv"],
      ["全量建筑矢量", "results/paper_strict/building_height_final.gpkg"],
      ["先验后验审计", "results/paper_strict/floor_prior_posthoc_audit.csv"],
      ["严格结果图", "results/figures/（6 张 SVG）"],
      ["论文版式图", "results/paper_figure_reproduction/（14 张 SVG）"],
      ["补充论文图", "results/footprint_constrained_paper/（8 张 SVG）"],
      ["机器检查", "inventory/final_reproduction_validation.json"],
    ],
    [3000, 6506],
  ),
  heading("报告口径", 2),
  body("本文档面向方法与结果理解，省略了大部分命令行参数和中间文件细节。完整参数、论文与代码差异及机器审计记录见 docs/00—05 系列文档。", { noIndent: true }),
);

const doc = new Document({
  creator: "Codex",
  title: "天津建筑三维重建论文复现总结报告（通俗版）",
  subject: "GAMMA-MCF 与 SBAS 建筑高度反演方法及结果总结",
  description: "基于真实复现结果编写的通俗版 Word 报告",
  styles: {
    default: { document: { run: { font, size: 22, color: colors.dark } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font, size: 46, bold: true, color: colors.navy }, paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 300 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font, size: 32, bold: true, color: colors.navy }, paragraph: { spacing: { before: 300, after: 180 }, outlineLevel: 0, keepNext: true, border: { bottom: { style: BorderStyle.SINGLE, size: 8, color: colors.teal, space: 4 } } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font, size: 27, bold: true, color: colors.blue }, paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Caption", name: "Caption", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font, size: 19, italic: true, color: colors.gray }, paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 180 } } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "○", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 300 } } } },
      ] },
      { reference: "steps", levels: [
        { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 360 } } } },
      ] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1120, right: 1200, bottom: 1120, left: 1200, header: 520, footer: 520 },
      },
    },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "B8C8D8", space: 2 } }, children: [run("天津建筑三维重建论文复现", { size: 18, color: colors.gray })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run("第 ", { size: 18, color: colors.gray }), new TextRun({ children: [PageNumber.CURRENT], font, size: 18, color: colors.gray }), run(" 页", { size: 18, color: colors.gray })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(outputPath, buffer);
  process.stdout.write(`${outputPath}\n`);
});
