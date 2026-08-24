%% 天津市区高分辨率SAR仿真 - 专业级侧视成像模拟
clear; clc; close all;

%% ========================
%% 1. 参数设置
%% ========================
rows = 4500;     % 距离向像素数
cols = 6000;     % 方位向像素数
pixel_size = 1;  % 空间分辨率 (1米/像素)

% SAR系统参数
wavelength = 0.031;      % L波段波长 (3.1 cm)
inc_angle = 30;          % 中心入射角 (度)
slant_range = 800000;    % 中心斜距 (800 km)
baseline = 150;          % 空间基线 (150 m)
h_platform = 700000;     % 平台高度 (700 km)

% 地形参数
base_elevation = 10;     % 基础海拔高度 (米)

% 输出控制
display_center = false;   % 是否只显示中心区域
center_rows = 2000:3000; % 中心区域行范围
center_cols = 2500:4000; % 中心区域列范围

fprintf('=== 天津市区SAR仿真开始 ===\n');
fprintf('影像尺寸: %d×%d (%.1f km×%.1f km)\n', rows, cols, cols/1000, rows/1000);
fprintf('系统参数: 波长=%.3fm, 入射角=%d°, 基线=%dm\n', wavelength, inc_angle, baseline);

%% ========================
%% 2. 地形建模 (DEM生成)
%% ========================
fprintf('生成DEM...');
tic;

% 创建坐标网格
[X, Y] = meshgrid(1:cols, 1:rows);

% 基础地形 (平坦城市)
dem_base = base_elevation + ...      % 基础海拔
           2 * sin(X/3000) + ...    % 轻微起伏
           1.5 * cos(Y/2500);

% 建筑物参数: [中心X, 中心Y, 宽度, 长度, 高度, 边缘锐度, 朝向]
% 朝向: 0=平行于方位向, 90=垂直于方位向
building_params = [
    % 市中心商业区 (密集高层)
    cols*0.4, rows*0.5, 80, 120, 250, 8, 0;
    cols*0.45, rows*0.52, 70, 90, 280, 8, 10;
    cols*0.38, rows*0.48, 60, 80, 220, 8, -5;
    
    % 金融区 (超高层)
    cols*0.6, rows*0.7, 100, 150, 350, 10, 0;
    cols*0.65, rows*0.72, 80, 100, 320, 10, 15;
    
    % 住宅区 (中等高度)
    cols*0.3, rows*0.8, 120, 180, 150, 6, 30;
    cols*0.25, rows*0.75, 100, 120, 130, 6, -20;
    
    % 工业区 (低矮建筑)
    cols*0.7, rows*0.3, 200, 300, 80, 4, 45;
    cols*0.8, rows*0.25, 150, 250, 70, 4, -30;
    
    % 特殊地标 (高塔)
    cols*0.5, rows*0.6, 30, 30, 400, 15, 0;
]; 

% 生成建筑模型
buildings = zeros(rows, cols);
building_mask = false(rows, cols);

for i = 1:size(building_params, 1)
    cx = building_params(i,1); cy = building_params(i,2);
    w = building_params(i,3); h = building_params(i,4); 
    z = building_params(i,5); sharp = building_params(i,6);
    orient = building_params(i,7); % 建筑物朝向
    
    % 旋转坐标以模拟建筑物朝向
    theta = deg2rad(orient);
    rotX = (X - cx)*cos(theta) - (Y - cy)*sin(theta);
    rotY = (X - cx)*sin(theta) + (Y - cy)*cos(theta);
    
    % 创建带锐利边缘的建筑模型
    building = z * exp(-(abs(rotX)/w).^sharp - (abs(rotY)/h).^sharp);
    
    % 添加到总建筑群
    buildings = buildings + building;
    building_mask = building_mask | (building > 0.1*z);
end

% 最终DEM
dem = dem_base + buildings;

fprintf('完成 (%.2fs)\n', toc);

%% ========================
%% 3. SAR侧视成像模拟
%% ========================
fprintf('模拟SAR侧视成像...');
tic;

% 计算地面距离和高度投影
ground_range = slant_range * cosd(inc_angle);
z0 = h_platform - ground_range * tand(inc_angle);

% 侧视成像几何模型 - 建筑物倾倒效应
% 关键公式: 投影位移 = (h * R) / (H * sinθ)
% 其中: h=建筑高度, R=斜距, H=平台高度, θ=入射角
proj_factor = slant_range / (h_platform * sind(inc_angle));

% 初始化侧视DEM
layover_dem = zeros(size(dem));
layover_effect = zeros(size(dem));

% 计算每个像素的投影位移
for y = 1:rows
    % 计算当前距离向的斜距
    r = slant_range + (y - rows/2) * pixel_size / sind(inc_angle);
    
    % 计算投影位移 (建筑物向近距方向倾倒)
    displacement = buildings(y, :) * proj_factor;
    
    % 计算投影后的位置 (位移为负表示向近距方向)
    proj_y = round(y - displacement);
    
    % 处理边界
    proj_y(proj_y < 1) = 1;
    proj_y(proj_y > rows) = rows;
    
    % 创建投影索引
    idx = sub2ind(size(dem), proj_y, 1:cols);
    
    % 在投影位置存储DEM值
    layover_dem(idx) = max(layover_dem(idx), dem(y, :));
    
    % 存储位移量用于相位计算
    layover_effect(y, :) = displacement;
end

% 填充未投影区域 (低矮地形)
for y = 1:rows
    for x = 1:cols
        if layover_dem(y, x) == 0
            layover_dem(y, x) = dem_base(y, x);
        end
    end
end

fprintf('完成 (%.2fs)\n', toc);

%% ========================
%% 4. 干涉相位生成
%% ========================
fprintf('生成干涉相位...');
tic;

% 相位计算公式: φ = (4π * B_⊥ * Δh) / (λ * R * sinθ)
% 其中: Δh = 高程变化, B_⊥ = 垂直基线
B_perp = baseline * cosd(inc_angle);

% 基础相位
phase = (4 * pi * B_perp * layover_dem) ./ (wavelength * slant_range * sind(inc_angle));

% 添加城市噪声特征
% 建筑物区域增加相位噪声 (失相关)
building_mask_layover = (layover_dem - dem_base) > 5; % 识别投影后的建筑物
phase_noise = 1.5 * randn(rows, cols) .* building_mask_layover; 
phase = phase + phase_noise;

% 添加轻微沉降信号
city_center_mask = exp(-((X - cols*0.4)/1200).^2 - ((Y - rows*0.5)/1200).^2);
subsidence = 0.5 * sin(X/1800 + Y/2000) .* city_center_mask;
phase = phase + subsidence;

% 相位缠绕 (wrapping)
diff = angle(exp(1i*phase));

fprintf('完成 (%.2fs)\n', toc);

%% ========================
%% 5. 强度图像生成
%% ========================
fprintf('生成强度图像...');
tic;

% 基础后向散射
mli_base = 0.4 + 0.2 * randn(rows, cols);

% 建筑物强反射效应
% 1. 立面反射 (与入射角相关)
building_facade = zeros(size(dem));
for y = 1:rows
    % 计算当前行的局部入射角
    local_inc = inc_angle + 5 * (y/rows - 0.5);
    
    % 建筑立面产生的强反射
    facade_strength = 0.7 * sind(local_inc) .* building_mask_layover(y, :);
    building_facade(y, :) = facade_strength;
end

% 2. 屋顶反射
building_roof = 0.4 * building_mask_layover;

% 3. 边缘角反射器效应
building_edges = edge(building_mask_layover, 'Canny', 0.15);
edge_strength = 1.2 * building_edges;

% 道路系统 (弱反射)
road_width = 15;
main_roads = (mod(X, 200) < road_width | (mod(Y, 200) < road_width));
road_strength = 0.25 * main_roads;

% 组合强度
mli = mli_base + building_facade + building_roof + edge_strength - road_strength;

% 添加散斑噪声 (乘性)
speckle = 0.3 * exp(randn(size(mli)) - 1);
mli = abs(mli .* (1 + speckle));

% 归一化
mli = mli - min(mli(:));
mli = mli / max(mli(:));

fprintf('完成 (%.2fs)\n', toc);

%% ========================
%% 6. 专业级可视化
%% ========================
fprintf('生成可视化...');
tic;

% 选择显示区域
if display_center
    disp_rows = center_rows;
    disp_cols = center_cols;
    disp_str = '中心区域';
else
    disp_rows = 1:rows;
    disp_cols = 1:cols;
    disp_str = '全区域';
end

% 创建专业SAR显示窗口
fig = figure('Position', [100, 100, 1400, 800], 'Name', '天津市区SAR仿真结果', 'Color', 'white');

% 1. DEM高程图 (原始地形)
subplot(2,3,1)
imagesc(disp_cols, disp_rows, dem(disp_rows, disp_cols));
title('原始DEM地形');
colormap(parula); colorbar;
axis equal tight; grid on;
xlabel('方位向 (像素)'); ylabel('距离向 (像素)');
set(gca, 'FontSize', 9);

% 2. 侧视投影DEM
subplot(2,3,2)
imagesc(disp_cols, disp_rows, layover_dem(disp_rows, disp_cols));
title('侧视投影DEM (建筑物倾倒)');
colormap(parula); colorbar;
axis equal tight; grid on;
xlabel('方位向 (像素)'); ylabel('距离向 (像素)');
set(gca, 'FontSize', 9);

% 3. 干涉相位
subplot(2,3,3)
imagesc(disp_cols, disp_rows, diff(disp_rows, disp_cols));
title('干涉相位 (缠绕)');
colormap(phasemap(256)); 
colorbar;
axis equal tight; grid on;
xlabel('方位向 (像素)'); ylabel('距离向 (像素)');
set(gca, 'FontSize', 9);

% 4. 强度图像 (MLI)
subplot(2,3,4)
imagesc(disp_cols, disp_rows, mli(disp_rows, disp_cols));
title('SAR强度图像');
colormap(gray); colorbar;
axis equal tight; grid on;
xlabel('方位向 (像素)'); ylabel('距离向 (像素)');
set(gca, 'FontSize', 9);

% 5. 建筑物位移量
subplot(2,3,5)
displ_vis = layover_effect(disp_rows, disp_cols);
displ_vis(displ_vis < 0.5) = NaN; % 只显示显著位移
imagesc(disp_cols, disp_rows, displ_vis);
title('建筑物位移量 (像素)');
colormap(jet); colorbar;
axis equal tight; grid on;
xlabel('方位向 (像素)'); ylabel('距离向 (像素)');
set(gca, 'FontSize', 9);

% 6. 3D SAR视图
subplot(2,3,6)
step = 10; % 降采样步长
surf(X(disp_rows(1:step:end), disp_cols(1:step:end), ...
     layover_dem(disp_rows(1:step:end), disp_cols(1:step:end)), ...
     'FaceColor', 'interp', 'EdgeColor', 'none'));
title('SAR 3D视图 (侧视几何)');
view(25, 45); 
light('Position', [-1 -1 0.5], 'Style', 'infinite');
lighting gouraud;
colormap(jet); colorbar;
xlabel('方位向 (像素)'); ylabel('距离向 (像素)'); zlabel('高程 (米)');
grid on; set(gca, 'FontSize', 9);

% 添加标题
sgtitle(sprintf('天津市区SAR仿真 - 分辨率: %dm | 比例尺: 1:%.0f', ...
        pixel_size, max(cols,rows)/500), 'FontSize', 14, 'FontWeight', 'bold');

fprintf('完成 (%.2fs)\n', toc);

%% ========================
%% 7. 数据保存
%% ========================
fprintf('保存数据...');
tic;

% 保存仿真数据
save('Tianjin_SAR_Simulation.mat', ...
     'dem', 'layover_dem', 'diff', 'mli', ...
     'building_mask', 'building_mask_layover', ...
     'pixel_size', 'wavelength', 'inc_angle', ...
     'baseline', 'slant_range', '-v7.3');

% 保存GeoTIFF格式 (可选)
% geotiffwrite('DEM_Original.tif', dem, R);
% geotiffwrite('SAR_Intensity.tif', mli, R);

fprintf('完成 (%.2fs)\n', toc);

%% ========================
%% 完成
%% ========================
fprintf('\n=== 仿真完成 ===\n');
fprintf('输出文件: Tianjin_SAR_Simulation.mat\n');
fprintf('建筑物数量: %d\n', size(building_params, 1));
fprintf('最大建筑高度: %.1f米\n', max(building_params(:,5)));
fprintf('最大位移量: %.1f像素\n', max(layover_effect(:)));