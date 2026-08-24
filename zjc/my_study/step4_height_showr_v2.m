% 创建 height_mask 矩阵
height_mask = zeros(size(data_rmli)); % 初始化为与 data_rmli 大小相同的全零矩阵

% 遍历每个 island，将其 HeightMatrix2 矩阵填充到 height_mask 中
for i = 1:numel(islands)
    % 提取当前 island 的 HeightMatrix2
    heightMatrix2 = islands(i).HeightMatrix2;

    % 获取当前 island 的左上角坐标
    topLeftCorner = islands(i).TopLeftCorner;

    % 获取当前 island 的尺寸
    sizeOfIsland = islands(i).Size;

    % 将当前 island 的 HeightMatrix2 填充到 height_mask 中
    height_mask(topLeftCorner(1):(topLeftCorner(1)+sizeOfIsland(1)-1), ...
                topLeftCorner(2):(topLeftCorner(2)+sizeOfIsland(2)-1)) = heightMatrix2;
end
% 获取矩阵尺寸
[rows, cols] = size(height_mask);

% 计算中心点
center_row = (rows + 1) / 2;
center_col = (cols + 1) / 2;

% 创建结果矩阵
shrunk_mask = zeros(rows, cols);

% 缩放系数
scale = 0.6;

% 对每个像素进行缩放变换
for i = 1:rows
    for j = 1:cols
        % 计算相对于中心的坐标
        rel_row = i - center_row;
        rel_col = j - center_col;
        
        % 应用缩放
        src_row = center_row + rel_row / scale;
        src_col = center_col + rel_col / scale;
        
        % 检查源坐标是否在原始矩阵范围内
        if src_row >= 1 && src_row <= rows && src_col >= 1 && src_col <= cols
            % 使用双线性插值获取源坐标处的值
            row_floor = floor(src_row);
            row_ceil = min(row_floor + 1, rows);
            col_floor = floor(src_col);
            col_ceil = min(col_floor + 1, cols);
            
            % 计算权重
            w_row = src_row - row_floor;
            w_col = src_col - col_floor;
            
            % 双线性插值
            shrunk_mask(i, j) = (1-w_row)*(1-w_col)*height_mask(row_floor, col_floor)+w_row*(1-w_col)*height_mask(row_ceil, col_floor)+w_row*w_col*height_mask(row_ceil, col_ceil);
        end
    end
end
height_mask=shrunk_mask;
shifted_mask = zeros(rows, cols);
% 进行平移操作
% 右移50像素，上移30像素
for i = 1:rows-40
    for j = 1:cols-100
        shifted_mask(i, j+100) = height_mask(i+40, j);
    end
end
height_mask=shifted_mask;

rows = size(data_geo, 1);
columns = size(data_geo, 2);
filename_rmli = '/media/mlxg/d/zjc/20240422.mli';

% 读取RMLI文件的数据
applyLog = true;
[data_rmli, count] = freadbkB(filename_rmli, lines, 'float32');

if applyLog
    offset = 1e3; % 可根据数据分布调整,调整底图明亮程度
    data_rmli = log10(data_rmli + 1e-10+offset); % 加10^-10以防止log(0)的情况
end

% 创建图形窗口，并设置大小
figure('Position', [100, 100, 800, 600]); % 800x600 像素的图形窗口
% 创建底图坐标系
ax1 = axes;
imagesc(ax1, data_rmli);
colormap(ax1, gray); % 底图使用灰度色图
axis(ax1, 'image');
ax1.Visible = 'on'; % 显示坐标轴
grid on;
xlabel('Column Index');
ylabel('Row Index');
title('Height Overlay on Grayscale Background');
hold on;

% 提取非零高度值并计算范围
nonZeroMask = height_mask > 0;
nonZeroHeights = height_mask(nonZeroMask);
minH = min(nonZeroHeights);
maxH = max(nonZeroHeights);

% 创建高度层坐标系（覆盖在底图上）
ax2 = axes;
h = imagesc(ax2, height_mask); % 在 ax2 中绘制高度层
colormap(ax2, parula); % 高度层使用 parula 色图
caxis(ax2, [minH, maxH]); % 设置高度层的颜色范围
h.AlphaData = nonZeroMask; % 关键！非零区域不透明，零值透明
axis(ax2, 'image');
ax2.Visible = 'off'; % 隐藏高度层坐标轴

% 同步两个坐标系的位置和尺寸
linkprop([ax1, ax2], {'Position', 'XLim', 'YLim'});

% 添加颜色条（仅针对高度层）
hColorbar = colorbar(ax2); % 注意指定 ax2
hColorbar.Label.String = 'Height (m)';
hColorbar.Ticks = linspace(minH, maxH, 5); % 5个等分刻度
hColorbar.TickLabels = arrayfun(@(v) sprintf('%.2f', v), hColorbar.Ticks, 'UniformOutput', false);

hold off;