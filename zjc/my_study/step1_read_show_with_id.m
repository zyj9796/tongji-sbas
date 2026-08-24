% 第一步，将geocoding文件和底图mli文件读取,并通过筛选做出效果图
% 设置geocoding文件名
filename_geo = '/media/mlxg/d/zjc/my_study/data/DEM/20231007.dem';
% 设置总行数
lines_geo = 7000;
lines = 7000;
% 调用 freadbkB 函数读取指定范围的数据
[data_geo, count_geo] = freadbkB(filename_geo, lines_geo, 'float32');
% data_geo = data_geo(:, 1:6600);
% 显示读取的数据和数量
disp(['Number of elements geocoding file read: ', num2str(count_geo)]);

% 读取底图数据
% filename_rmli = 'H:\tianyi2\INTF\20231029-20240422.diff_flt';
filename_rmli = '/media/mlxg/d/zjc/tianyi2/INTF/ave';

applyLog = true;
[data_rmli, count] = freadbkB(filename_rmli, lines_geo, 'float32');
% 显示读取的数据和数量
disp(['Number of elements geocoding file read: ', num2str(count)]);
% data_rmli = data_rmli(:, 1:6600);
disp('reading finished!')

% 原始数据
rows = size(data_geo, 1);
columns = size(data_geo, 2);

% ==================== 主要修改开始 ====================
% 分离楼层信息（整数部分）和建筑物ID（小数部分）
floor_data = floor(data_geo);  % 整数部分：楼层信息
id_data = data_geo - floor_data;  % 小数部分：建筑物ID信息

% 由于插值可能导致ID有微小差异，四舍五入到4位小数以确保一致性
%id_data = round(id_data, 4);

% 获取唯一的建筑物ID
unique_ids = unique(id_data);
num_ids = length(unique_ids);
disp(['Number of unique building IDs detected: ', num2str(num_ids)]);

% 计算每个建筑物ID对应的楼层值（取众数）
id_floors = zeros(size(unique_ids));
for i = 1:length(unique_ids)
    mask_id = (id_data == unique_ids(i));
    floors_in_id = floor_data(mask_id);
    if ~isempty(floors_in_id)
        id_floors(i) = mode(floors_in_id);
    else
        id_floors(i) = 0;
    end
end

% 按楼层值从高到低排序，确保高楼层建筑物覆盖低楼层
[id_floors_sorted, sort_idx] = sort(id_floors, 'descend');
unique_ids_sorted = unique_ids(sort_idx);

% 初始化 mask_real 矩阵为 NaN
mask_real = nan(rows, columns);

% 设置扩展参数，单位为像素
expandLeft = 50;  % 向左扩展的像素数
expandRight = 20; % 向右扩展的像素数
expandUp = 10;    % 向上扩展的像素数
expandDown = 10;  % 向下扩展的像素数

% 遍历所有建筑物ID（按楼层从高到低）
for idx = 1:length(unique_ids_sorted)
    current_id = unique_ids_sorted(idx);
    current_floor = id_floors_sorted(idx);
    
    % 找到所有等于当前建筑物ID的像素位置
    [row_indices, col_indices] = find(id_data == current_id);
    
    % 对每个像素进行扩展（基于该建筑物的楼层值）
    for i = 1:length(row_indices)
        row = row_indices(i);
        col = col_indices(i);
        
        if current_floor > 30  % 检查数值大于30的条件
            % 计算各方向的扩展量（基于当前建筑物的楼层值）
            x = round(1.9 * current_floor);  % 原本的扩展量
            
            % 向左扩展
            left_bound = max(1, col - x - round(expandLeft * current_floor/70));
            % 向右扩展
            right_bound = min(columns, col + round(expandRight * current_floor/70));
            % 向上扩展
            up_bound = max(1, row - round(expandUp * current_floor/70));
            % 向下扩展
            down_bound = min(rows, row + round(expandDown * current_floor/70));
            
            % 对扩展区域内的mask_real赋值为当前建筑物的楼层值
            mask_real(round(up_bound):round(down_bound), round(left_bound):round(right_bound)) = current_floor;
        end
    end
end

% 更新最大楼层值（用于颜色条）
maxValue = max(floor_data(:));
% ==================== 主要修改结束 ====================

% 以下部分保持不变
shift_down = 10; % 向下移动 2 像素
shift_right = 35; % 向右移动 10 像素

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

% 创建图形窗口，并设置大小
figure('Position', [100, 100, 800, 600]); % 800x600 像素的图形窗口
% 显示 data_rmli 图像
if applyLog
    offset = 1e3; % 可根据数据分布调整,调整底图明亮程度
    data_rmli = log10(data_rmli + offset);
end
imagesc(data_rmli); % 使用 imagesc 函数绘制数据
colormap('gray'); % 背景使用灰色调，以突出叠加的多彩掩膜
% colorbar; % 添加颜色条
% 保持添加新的图层
hold on;
% 使用 jet colormap 生成彩色图像
jetColors = jet(256); % 生成 jet colormap
maskNormalized = (mask_real - min(mask_real(:))) / (max(mask_real(:)) - min(mask_real(:)));
maskNormalized = round(maskNormalized * 255) + 1; % 将数据值映射到 [1, 256]
coloredImage = ind2rgb(maskNormalized, jetColors);

% 显示图像
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
title('Building islands (with ID-based separation)');
hold off;

% 创建独立的 colorbar
figure('Position', [100, 100, 200, 600]); % 设置窗口大小和位置
colormap('jet'); % 使用 jet 颜色映射
caxis([0 maxValue]); % 设置颜色条范围，与建筑物层数对应
colorbar; % 添加颜色条
ylabel(colorbar, 'Building Levels'); % 为颜色条添加标签
title('Colorbar for Building Levels'); % 设置标题