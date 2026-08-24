%并行池，感觉没啥用
%poolobj = parpool('local', 48);
lines=2500;
disp('Unwrapping the interferogram!'); % 开始解缠干涉图像
aven = 'H:\TY_tianjing\INTF\ave'; % 平均振幅文件路径
mag = freadbkB(aven, lines, 'float32'); % 读取浮点型数据
% 构造用于存储距离向和方位向的数组，xazis（i）,xranges(i)表示第i个像素的雷达坐标
[xazis, xranges] = find(ones(size(mag)));
mag_db = mag2db(mag);

N_knn = 9; % 设定KNN的邻居数
accu = 1; % 精度，单位为毫米
diff = sqrt(2) * accu; % 用于相位差异计算
c = 299792458; % 光速
wavelen = 1000 * c / 5.4050005e+09; % 波长计算
thre_accu = (4 * pi) * diff / wavelen; % 精度阈值计算

inft = 'H:\TY_tianjing\INTF\20231007-20231029.diff'; % 干涉图文件路径
image = freadbkB(inft, 2500, 'cpxfloat32'); % 读取复数浮点类型数据
phases = angle(image); % 获取相位信息

[rows, cols] = size(phases); % 获取图像的尺寸
phase_vec = phases(:); 
indxx = find(ones(size(phases)));
[azis, ranges] = find(ones(size(phases))); % 获取所有像素的方位和距离
% 构造候选观测集合
obs_candi = [indxx, ranges, azis, phase_vec];

% 删除强度不显著的像元
nn = mag_db < 90;
obs_candi(nn, :) = [];

disp('Start estimation on the full image'); % 开始在整个图像上估计
pwhole = [];
resid_keep = [];

% 在完整图像上计算弧段
ranges = obs_candi(:, 2);
azis = obs_candi(:, 3);
obs_phase = obs_candi(:, end);

np = [ranges, azis];
% 进行局部K近邻搜索以形成弧段
IDX = local_knn(np, N_knn, 0);
IDX_froml = IDX.from;
IDX_tol = IDX.to;
narc = length(IDX_froml); % 弧段数量
IDX_from = obs_candi(IDX_froml, 1);
IDX_to = obs_candi(IDX_tol, 1);

a = 0;
b = 1;

% 归一化范围和方位
m = min(ranges);
M = max(ranges);
ranges_n = (b - a) * (ranges - m) / (M - m);
m = min(azis);
M = max(azis);
azis_n = (b - a) * (azis - m) / (M - m);

% 计算弧段构造所需的差分
dr = ranges_n(IDX_tol) - ranges_n(IDX_froml);
dazi = azis_n(IDX_tol) - azis_n(IDX_froml);
drdazi = ranges_n(IDX_tol) .* azis_n(IDX_tol) - ranges_n(IDX_froml) .* azis_n(IDX_froml);
arc_poly = [dr', dazi', drdazi'];

% 准备观测数据
y = wrap(obs_phase(IDX_tol) - obs_phase(IDX_froml));

% 使用鲁棒拟合方法估计参数
[b, stats] = robustfit(arc_poly, y, 'bisquare', 4.685, 'off');
obse = arc_poly * b;

% 统计残差
res = stats.resid;

% 筛选满足精度要求的弧段
nn = find(abs(res) < thre_accu);
IDX_from_keep = IDX_from(nn);
IDX_to_keep = IDX_to(nn);
obse_keep = obse(nn, :);
res_keep = res(nn, :);

pwhole = [IDX_from_keep, IDX_to_keep, obse_keep];
resid_keep = [IDX_from_keep, IDX_to_keep, res_keep];

NARC = length(pwhole(:, 1)); % 弧段数量
NTCP = length(obs_candi(:, 1)); % 总控制点数

% 构建与弧段相关的设计矩阵
arc_index = (1:NARC)';
arc_tcp1 = sparse(arc_index, pwhole(:, 1), -1 * ones(NARC, 1), NARC, NTCP);
arc_tcp2 = sparse(arc_index, pwhole(:, 2), ones(NARC, 1), NARC, NTCP);
arc_tcp = arc_tcp1 + arc_tcp2;

clear arc_tcp1 arc_tcp2

% 选择非零的设计矩阵列
xxx = sum(abs(arc_tcp), 1);
xxxd = find(xxx ~= 0);
tempd = arc_tcp(:, xxxd);   
tempd(:, 1) = [];

% 求解相对行为
par_re = lsmr(tempd, pwhole(:, end));
par_re = [0; par_re];
resi_re = lsmr(tempd, resid_keep(:, end));
resi_re = [0; resi_re];

% 结果存储到最终输出数组中
xxx = nan(size(mag_db));
resi_xxx = xxx;
gind = obs_candi(:, 1);
xxx(gind(xxxd)) = par_re;
resi_xxx(gind(xxxd)) = resi_re;
unw_ = xxx;
residual_ = resi_xxx;
disp('Unwrapping finished!'); % 开始解缠干涉图像
