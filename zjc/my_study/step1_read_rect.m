%第一步，将geocoding文件和底图bmp文件读取,并通过筛选做出效果图
% 设置geocoding文件名
filename_geo = '/media/mlxg/d/zjc/my_study/data/DEM/20231007.dem';
% 设置总行数
lines_geo = 7000;
lines = 7000;

% 调用 freadbkB 函数读取指定范围的数据
[data_geo, count_geo] = freadbkB(filename_geo, lines_geo, 'float32');
% 显示读取的数据和数量
disp(['Number of elements geocoding file read: ', num2str(count_geo)]);

% 读取BMP底图数据
filename_bmp = '/media/mlxg/d/zjc/tianyi2/INTF/ave.bmp';
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
    for channel = 1:3
        filtered_image(:,:,channel) = imgaussfilt(data_rmli(:,:,channel), sigma);
    end
else  % 灰度图像
    filtered_image = imgaussfilt(data_rmli, sigma);
end
%标记
disp('Gaussian filtering completed.');
filtered_image = data_rmli;
% 原始数据
rows = size(data_geo, 1);
columns = size(data_geo, 2);

% 初始化 mask_real 矩阵为 NaN
mask_real = nan(rows, columns);

% 找到数据中的最大值
maxValue = max(data_geo(:));

% 设置扩展参数，单位为像素
expandLeft = 30;  % 向左扩展的像素数
expandRight = 20; % 向右扩展的像素数
expandUp = 0;    % 向上扩展的像素数
expandDown = 0;  % 向下扩展的像素数

for value = 1:1:maxValue
    % 找到所有等于当前值的像素位置
    [row_indices, col_indices] = find(data_geo == value);
   
    % 对每个找到的像素进行扩展
    for i = 1:length(row_indices)
        row = row_indices(i);
        col = col_indices(i);
        if value > 30  % 检查数值大于30的条件
            % 计算各方向的扩展量
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
% 新功能：识别连通区域并绘制红色矩形框
% ============================================

% 创建图形窗口，并设置大小
figure('Position', [100, 100, 800, 600]);

% 显示滤波后的BMP底图
imshow(filtered_image, []);
colormap('gray');
hold on;

% 获取所有唯一的楼层值（大于30的）
unique_values = unique(mask_real(:));
unique_values = unique_values(unique_values > 30);

disp(['Found ', num2str(length(unique_values)), ' unique building levels above threshold']);

% 用于存储所有矩形框的坐标
all_rectangles = [];

% 对每个楼层值进行连通区域分析
for level_idx = 1:length(unique_values)
    current_level = unique_values(level_idx);
    
    % 创建当前楼层的二值掩膜
    binary_mask = (mask_real == current_level);
    
    if any(binary_mask(:))
        % 使用4连通性进行连通区域标记
        cc = bwconncomp(binary_mask, 4);
        
        % 获取每个连通区域的属性
        stats = regionprops(cc, 'BoundingBox');
        
        % 为每个连通区域绘制矩形框
        for region_idx = 1:length(stats)
            bbox = stats(region_idx).BoundingBox;
            
            % 调整边界框坐标（regionprops返回的是[x, y, width, height]）
            x = bbox(1);
            y = bbox(2);
            width = bbox(3);
            height = bbox(4);
            
            % 存储矩形信息
            all_rectangles = [all_rectangles; x, y, width, height, current_level];
            
            % 绘制红色矩形框
            rectangle('Position', [x, y, width, height], ...
                     'EdgeColor', 'r', ...
                     'LineWidth', 1, ...
                     'LineStyle', '-');
            
            % 可选：在矩形左上角显示楼层数
            text(x-5, y-15, num2str(current_level), ...
                'Color', 'y', ...
                'FontSize', 12, ...
                'FontWeight', 'bold');
        end
        
        disp(['Level ', num2str(current_level), ': found ', ...
              num2str(length(stats)), ' connected regions']);
    end
end

% 显示坐标轴和网格
axis on;
axis image;
grid on;

% 设置图像标签
xlabel('Column Index');
ylabel('Row Index');
title('Building Islands - Red Rectangle Detection (4-connectivity)');
hold off;

% ============================================
% 显示统计信息
% ============================================
disp('=== Processing Summary ===');
disp(['Total building levels above 30: ', num2str(length(unique_values))]);
disp(['Total rectangles drawn: ', num2str(size(all_rectangles, 1))]);
disp(['Building level range: ', num2str(min(unique_values)), ...
      ' to ', num2str(max(unique_values))]);

% 可选：显示矩形框信息表格
if ~isempty(all_rectangles)
    disp('=== Rectangle Information ===');
    disp('   X     Y     Width  Height  Level');
    for i = 1:min(10, size(all_rectangles, 1))  % 显示前10个
        fprintf('%5.0f %5.0f %6.0f %6.0f %6.0f\n', all_rectangles(i,:));
    end
    if size(all_rectangles, 1) > 10
        disp(['... and ', num2str(size(all_rectangles, 1)-10), ' more rectangles']);
    end
end

disp('Processing completed successfully!');