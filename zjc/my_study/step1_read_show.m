%第一步，将geocoding文件和底图bmp文件读取,并通过筛选做出效果图
% 设置geocoding文件名
filename_geo = '/media/mlxg/d/zjc/my_study/数据/DEM/20231007.dem';
% 设置总行数
lines_geo = 7000;
lines = 7000;

% 调用 freadbkB 函数读取指定范围的数据
[data_geo, count_geo] = freadbkB(filename_geo, lines_geo, 'float32');
% 显示读取的数据和数量
disp(['Number of elements geocoding file read: ', num2str(count_geo)]);

% 读取BMP底图数据
filename_bmp = '/media/mlxg/d/zjc/tianyi2/INTF/mli/ave.bmp';
disp(['Reading BMP image: ', filename_bmp]);

% 读取BMP图像
data_rmli = imread(filename_bmp);
disp(['BMP image size: ', num2str(size(data_rmli, 1)), ' x ', ...
      num2str(size(data_rmli, 2)), ' x ', num2str(size(data_rmli, 3))]);

% ============================================
% 应用高斯滤波（方案B）
% ============================================
sigma = 2;  % 标准差，值越大越平滑
disp(['Applying Gaussian filter with sigma = ', num2str(sigma), '...']);

% 对BMP图像应用高斯滤波
if size(data_rmli, 3) == 3  % RGB图像
    filtered_image = zeros(size(data_rmli));
    for i = 1:3
        filtered_image(:,:,i) = imgaussfilt(data_rmli(:,:,i), sigma);
    end
else  % 灰度图像
    filtered_image = imgaussfilt(data_rmli, sigma);
end

data_rmli = filtered_image;
disp('Gaussian filtering completed.');

% 原始数据
rows = size(data_geo, 1);
columns = size(data_geo, 2);

% 初始化 mask_real 矩阵为 NaN
mask_real = nan(rows, columns);

% 找到数据中的最大值
maxValue = max(data_geo(:));

% 从最大值-1进行操作
% 设置扩展参数，单位为像素
expandLeft = 10;  % 向左扩展的像素数
expandRight = 10; % 向右扩展的像素数
expandUp = 10;    % 向上扩展的像素数
expandDown = 10;  % 向下扩展的像素数
data_geo=round(data_geo);
for value = 1:1:maxValue
    % 找到所有等于当前值的像素位置
    [row_indices, col_indices] = find(data_geo == value);
   
    % 对每个找到的像素进行扩展
    for i = 1:length(row_indices)
        row = row_indices(i);
        col = col_indices(i);
        if value > 40 %& value < 30 % 检查数值大于30的条件
            % 计算各方向的扩展量（包括原本的扩展量与额外的设置）
            x = round(1.9 * value);  % 原本的扩展量
            
            % 向左扩展
            left_bound = max(1, col - x - round(expandLeft* value/70));
            % 向右扩展
            right_bound = min(size(data_geo, 2), col  + round(expandRight* value/70));
            % 向上扩展
            up_bound = max(1, row  - round(expandUp* value/70));
            % 向下扩展
            down_bound = min(size(data_geo, 1), row  + round(expandDown* value/70));
            
            % 对扩展区域内的mask_real赋值为当前的value
            mask_real(round(up_bound):round(down_bound), round(left_bound):round(right_bound)) = value;
        end
    end
end

shift_down = 10; % 向下移动 10 像素
shift_right = 35; % 向右移动 35 像素

% 创建与 mask_real 同大小、初值为 0 的新矩阵
mask_real_shifted = zeros(size(mask_real), 'like', mask_real);

% 将原先 mask_real 的内容"粘贴"到新矩阵相应位置
% 注意这里要确保索引不会越界
rows = size(mask_real, 1);
cols = size(mask_real, 2);
if rows > shift_down && cols > shift_right
    mask_real_shifted(1+shift_down:end, 1+shift_right:end) = ...
        mask_real(1:end-shift_down, 1:end-shift_right);
end

% 将移动后的矩阵重新赋值给 mask_real
mask_real = mask_real_shifted;

% 将 NaN 替换为 0
mask_real(isnan(mask_real)) = 0;

% ============================================
% 显示结果
% ============================================

% 创建图形窗口，并设置大小
figure('Position', [100, 100, 800, 600]); % 800x600 像素的图形窗口

% 显示滤波后的BMP底图
if size(data_rmli, 3) == 3  % RGB图像
    imshow(data_rmli);
else  % 灰度图像
    imshow(data_rmli, []);
    colormap('gray');
end

% 保持添加新的图层
hold on;

% 使用 jet colormap 生成彩色图像
jetColors = jet(256); % 生成 jet colormap
maskNormalized = (mask_real - min(mask_real(:))) / (max(mask_real(:)) - min(mask_real(:)));
maskNormalized = round(maskNormalized * 255) + 1; % 将数据值映射到 [1, 256]
coloredImage = ind2rgb(maskNormalized, jetColors);

% 显示建筑物图层
img_mask = imshow(coloredImage);

% 设置透明度，mask_real 为 0 的地方透明
set(img_mask, 'AlphaData', mask_real > 0);

% 显示坐标轴和网格
axis on;
axis image;
grid on;

% 设置图像标签
xlabel('Column Index');
ylabel('Row Index');
title('Building Islands Overlay on Filtered BMP Image (Gaussian Filter)');
hold off;

% 创建独立的 colorbar
figure('Position', [100, 100, 200, 600]); % 设置窗口大小和位置
colormap('jet'); % 使用 jet 颜色映射
caxis([0 maxValue]); % 设置颜色条范围，与建筑物层数对应
colorbar; % 添加颜色条
ylabel(colorbar, 'Building Levels'); % 为颜色条添加标签
title('Colorbar for Building Levels'); % 设置标题

% ============================================
% 显示统计信息
% ============================================
disp('=== Processing Summary ===');
disp(['Total pixels: ', num2str(rows * columns)]);
disp(['Pixels above threshold (30): ', num2str(sum(mask_real(:) > 0))]);
disp(['Building level range: ', num2str(min(mask_real(mask_real>0))), ...
      ' to ', num2str(maxValue)]);
disp('Processing completed successfully!');