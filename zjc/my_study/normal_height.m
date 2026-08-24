% 定义文件名和路径
filename = '/media/mlxg/d/zjc/tianyi2/INTF/20240525-20240605.diff_flt';
lines = 7000; % 原图像总行数
filename_cc = '/media/mlxg/d/zjc/tianyi2/INTF/20240525_20240605.cc';

% 读取复数数据
[data, count] = freadbkB(filename, lines, 'cpxfloat32');
[data_cc, count2] = freadbkB(filename_cc, lines, 'float32');

% 显示读取信息
disp(['Number of elements read (complex data): ', num2str(count)]);
disp(['Number of elements read (coherence): ', num2str(count2)]);

% 确定原图像尺寸
[rows, cols] = size(data);
disp(['Original image size: ', num2str(rows), ' x ', num2str(cols)]);

% 定义从左上角开始的截取位置和尺寸
startRow = 1000;  % 起始行索引
startCol = 2000;  % 起始列索引
targetRows = 500; % 截取的行数
targetCols = 1000; % 截取的列数

% 确保截取范围在图像范围内
if startRow + targetRows - 1 > rows
    targetRows = rows - startRow + 1;
    warning('调整截取行数以避免越界');
end
if startCol + targetCols - 1 > cols
    targetCols = cols - startCol + 1;
    warning('调整截取列数以避免越界');
end

% 截取指定区域的复数数据和相干性图
croppedData = data(startRow:startRow+targetRows-1, startCol:startCol+targetCols-1);
croppedCC = data_cc(startRow:startRow+targetRows-1, startCol:startCol+targetCols-1); % 截取相同区域的相干性
%
%croppedData=data;
%croppedCC=data_cc;
% 显示截取后的幅度图（复数数据的幅度）
croppedMag = abs(croppedData);
figure;
set(gcf, 'Position', [100, 100, 800, 600]);
imagesc(croppedMag);
colormap('jet');
colorbar;
xlabel('Column Index');
ylabel('Row Index');
title(['Cropped Magnitude (From [', num2str(startRow), ',', num2str(startCol), '] Size ', num2str(targetRows), 'x', num2str(targetCols), ')']);

% 显示截取后的相位图（复数数据的相位）
croppedPhase = angle(croppedData);
figure;
set(gcf, 'Position', [100, 100, 800, 600]);
imagesc(croppedPhase);
colormap('jet');
colorbar;
xlabel('Column Index');
ylabel('Row Index');
title(['Cropped Phase (From [', num2str(startRow), ',', num2str(startCol), '] Size ', num2str(targetRows), 'x', num2str(targetCols), ')']);

% 显示截取的相干性图
figure;
set(gcf, 'Position', [100, 100, 800, 600]);
imagesc(croppedCC);
colormap('jet');
colorbar;
%clim([0 1]); % 设置颜色轴范围0-1，因为相干性系数在此范围内
xlabel('Column Index');
ylabel('Row Index');
title(['Cropped Coherence (From [', num2str(startRow), ',', num2str(startCol), '] Size ', num2str(targetRows), 'x', num2str(targetCols), ')']);

% 根据相干性阈值生成二值掩膜
coherence_threshold = 0.3; % 这是一个关键参数，需要根据您的数据特性进行调整。通常尝试0.2, 0.3, 0.4...
mask = croppedCC > coherence_threshold;
    
% 显示生成的掩膜
figure;
set(gcf, 'Position', [100, 100, 800, 600]);
imagesc(mask);
colormap('gray');
colorbar;
xlabel('Column Index');
ylabel('Row Index');
title(['Binary Mask (Coherence > ', num2str(coherence_threshold), ')']);

% 进行相位解缠 - 使用Goldstein算法并传入复数数据和相干性掩膜
unwrappedPhase = GoldsteinUnwrap2D(croppedData,mask);
% 显示解缠后的相位

figure;
set(gcf, 'Position', [100, 100, 800, 600]);

% 创建一个副本用于显示，避免修改原始数据
displayPhase = unwrappedPhase;

% 将掩膜区域（值为0）设置为NaN，这样imagesc会自动将其显示为黑色[1](@ref)
displayPhase(displayPhase == 0) = NaN;

% 显示相位数据
imagesc(displayPhase);

% 创建并应用自定义colormap：从黑色开始，然后是jet colormap[6,7](@ref)
% 这样确保值为NaN的区域显示为黑色，其他区域使用jet颜色映射
customMap = [0 0 0; jet(255)]; % 黑色 + 标准的jet颜色映射
colormap(customMap);

% 设置颜色轴范围，确保所有有效数据正确映射到颜色[5](@ref)
% 找到非零（即有效）数据的最小值和最大值
validData = unwrappedPhase(unwrappedPhase ~= 0);
if ~isempty(validData)
    caxis([min(validData) max(validData)]);
end

colorbar;
xlabel('Column Index');
ylabel('Row Index');
title('Unwrapped Phase (Goldstein Algorithm with Coherence Mask)');

% 保存当前显示的图像（即修改后的图像）[5](@ref)
% 获取当前帧对应的图像数据
frame = getframe(gcf);
img = frame2im(frame);
% 保存图像
imwrite(img, '/media/mlxg/d/zjc/tianyi2/INTF/20240525-20240605_unwrappedPhase.png');
heightMatrix = phaseToElevation(unwrappedPhase,[1,1]);
% 假设您的数据已经准备好
% heightMatrix = phaseToElevation(unwrappedPhase,[1,1]);

%% 1. 创建精美的伪彩色图 (2D Heatmap)
figure('Position', [100, 100, 800, 600], 'Color', 'w'); % 设置图形位置和背景色
h_image = imagesc(heightMatrix);

% 设置颜色映射 - 地形色带通常更直观
colormap(gca, 'terrain'); % 或尝试 'parula', 'hot', 'jet', 'gray'
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
colormap(gca, 'terrain');
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
colormap(gca, 'terrain');
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
colormap(gca, 'terrain');
shading interp;
hold on;
% 在底部绘制等高线投影
contour3(X, Y, Z, 15, 'LineColor', 'k', 'LineWidth', 1);
hold off;
view(-30, 45);
axis tight;
y
% 添加光照和美化（同方案2）
light('Position', [1, 0, 1], 'Style', 'infinite');
lighting gouraud;
h_surf_combined.AmbientStrength = 0.3;
h_surf_combined.DiffuseStrength = 0.7;
% ... (其余美化代码同方案2)

title('3D Terrain with Contour Lines', 'FontSize', 14, 'FontWeight', 'bold');
% ... (其余标签、颜色条等代码同方案2)