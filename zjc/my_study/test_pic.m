
%% 1. 创建精美的伪彩色图 (2D Heatmap)
figure('Position', [100, 100, 800, 600], 'Color', 'w'); % 设置图形位置和背景色
h_image = imagesc(heightMatrix);

% 设置颜色映射 - 地形色带通常更直观
colormap(gca, 'jet'); % 或尝试 'parula', 'hot', 'jet', 'gray'
colorbar_label = colorbar;
ylabel(colorbar_label, 'Elevation (m)', 'FontSize', 11, 'FontWeight', 'bold'); % 设置colorbar标签

% 美化坐标轴和标题
axis equal tight; % 保持比例并紧贴数据范围
title('Elevation Distribution', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);
set(gca, 'FontSize', 11, 'LineWidth', 1.5); % 设置坐标轴字体和线宽

% 可选：添加网格线以便更精确地读值
grid on;
set(gca, 'XMinorTick', 'on', 'YMinorTick', 'on', 'GridAlpha', 0.3);

%% 2. 创建带光照的三维曲面图 (3D Surface with Lighting)
figure('Position', [100, 100, 900, 700], 'Color', 'w');
% 减小数据量以获得更流畅的渲染效果（可选）
[rows, cols] = size(heightMatrix);
skip = max(floor(rows/500), 1); % 控制缩减因子
[X, Y] = meshgrid(1:skip:cols, 1:skip:rows);
Z = heightMatrix(1:skip:rows, 1:skip:cols);

h_surf = surf(X, Y, Z, 'EdgeColor', 'none'); % 'none' 隐藏网格线使表面更平滑
view(-30, 45); % 设置视角 (方位角, 仰角)
colormap(gca, 'jet');
shading interp; % 平滑颜色插值

% 添加光照以增强立体感
light('Position', [1, 0, 1], 'Style', 'infinite');
lighting gouraud; % 使用Gouraud着色算法，效果更平滑
h_surf.AmbientStrength = 0.3;
h_surf.DiffuseStrength = 0.7;
h_surf.SpecularStrength = 0.1;
h_surf.SpecularExponent = 25;

% 美化
title('3D Terrain Surface', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);
zlabel('Elevation (m)', 'FontSize', 12);
set(gca, 'FontSize', 11, 'LineWidth', 1.5, 'Box', 'on');
axis tight;

% 添加颜色条
colorbar_label_3d = colorbar;
ylabel(colorbar_label_3d, 'Elevation (m)', 'FontSize', 11, 'FontWeight', 'bold');

%% 3. 创建等高线图 (Contour Plot)
figure('Position', [100, 100, 800, 600], 'Color', 'w');
% 先绘制填充等高线图提供背景
[c, h] = contourf(heightMatrix, 15, 'LineColor', 'none'); % 15代表等高线数量
colormap(gca, 'jet');
hold on;
% 再绘制线性等高线图突出轮廓
contour(heightMatrix, 15, 'LineColor', [0.3, 0.3, 0.3], 'LineWidth', 0.5);
hold off;

colorbar;
title('Elevation Contour Map', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('X (pixels)', 'FontSize', 12);
ylabel('Y (pixels)', 'FontSize', 12);
set(gca, 'FontSize', 11, 'LineWidth', 1.5);

%% 4. (高级) 组合图：三维曲面 + 等高线
figure('Position', [100, 100, 900, 700], 'Color', 'w');
% 绘制曲面
h_surf_combined = surf(X, Y, Z, 'EdgeColor', 'none', 'FaceAlpha', 0.9);
colormap(gca, 'jet');
shading interp;
hold on;
% 在底部绘制等高线投影
contour3(X, Y, Z, 15, 'LineColor', 'k', 'LineWidth', 1);
hold off;
view(-30, 45);
axis tight;

% 添加光照和美化（同方案2）
light('Position', [1, 0, 1], 'Style', 'infinite');
lighting gouraud;
h_surf_combined.AmbientStrength = 0.3;
h_surf_combined.DiffuseStrength = 0.7;
% ... (其余美化代码同方案2)

title('3D Terrain with Contour Lines', 'FontSize', 14, 'FontWeight', 'bold');
% ... (其余标签、颜色条等代码同方案2)