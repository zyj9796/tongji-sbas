% ============================================
% SAR强度图显示方案（简单滤波优化版）
% ============================================

% 第一步：读取geocoding数据
filename_geo = '/media/mlxg/d/zjc/my_study/data/DEM/20231007.dem';
lines_geo = 7000;

[data_geo, count_geo] = freadbkB(filename_geo, lines_geo, 'float32');
disp(['Geocoding elements read: ', num2str(count_geo)]);

% 第二步：读取BMP底图
filename_bmp = '/media/mlxg/d/zjc/tianyi2/INTF/ave.bmp';
disp(['Reading BMP image: ', filename_bmp]);

% 读取BMP图像
base_image = imread(filename_bmp);
disp(['BMP image size: ', num2str(size(base_image, 1)), ' x ', ...
      num2str(size(base_image, 2)), ' x ', num2str(size(base_image, 3))]);

% ============================================
% 简单滤波方案（二选一）
% ============================================

% 方案A：中值滤波（推荐，适合SAR图像去噪）
disp('Applying median filter (3x3 window)...');
%filtered_image = medfilt2(base_image, [3 3]);

% 方案B：高斯滤波（备用，平滑效果更均匀）
sigma = 2;  % 标准差，值越大越平滑
filtered_image = imgaussfilt(base_image, sigma);

disp('Filtering completed.');

% 获取图像尺寸
rows = size(data_geo, 1);
columns = size(data_geo, 2);

% ============================================
% 核心处理：只保留30层以上的像素
% ============================================
mask_real = zeros(rows, columns);
threshold = 30;
mask_real(data_geo > threshold) = data_geo(data_geo > threshold);

% ============================================
% 显示结果（优化显示方式）
% ============================================

% 创建主显示窗口
figure('Position', [100, 100, 800, 600]);

% 显示滤波后的BMP底图（使用imshow自动对比度调整）
imshow(filtered_image, []);
colormap('gray');
hold on;

% 使用jet colormap生成建筑物彩色图像
jetColors = jet(256);
maskNormalized = (mask_real - min(mask_real(:))) / (max(mask_real(:)) - min(mask_real(:)));
maskNormalized = round(maskNormalized * 255) + 1;
coloredImage = ind2rgb(maskNormalized, jetColors);

% 显示建筑物图层
img_mask = imshow(coloredImage);

% 设置透明度：建筑物区域50%不透明度，背景完全透明
alpha_data = (mask_real > 0) * 0.9;
set(img_mask, 'AlphaData', alpha_data);

% 显示坐标轴和网格
axis on;
axis image;
grid on;

% 设置标签
xlabel('Column Index');
ylabel('Row Index');
title('Building Islands Overlay on Filtered BMP Image (Median Filter)');
hold off;

% ============================================
% 创建独立的colorbar
% ============================================
figure('Position', [100, 100, 200, 600]);
colormap('jet');
maxValue = max(mask_real(:));
caxis([0 maxValue]);
colorbar;
ylabel(colorbar, 'Building Levels');
title('Colorbar for Building Levels');

% ============================================
% 显示统计信息
% ============================================
disp('=== Processing Summary ===');
disp(['Total pixels: ', num2str(rows * columns)]);
disp(['Pixels above threshold (', num2str(threshold), '): ', ...
      num2str(sum(mask_real(:) > 0))]);
disp(['Building level range: ', num2str(min(mask_real(mask_real>0))), ...
      ' to ', num2str(maxValue)]);
disp('Processing completed successfully!');