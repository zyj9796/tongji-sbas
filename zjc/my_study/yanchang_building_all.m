% 假设 Z_shifted 已经存在并且是 250x250 的矩阵
% 例如:
% Z_shifted = randi([-1, 255], 250, 250); % 随机生成一个250x250的矩阵用于测试

% 初始化 Z_final 为 Z_shifted 的拷贝
Z_final = Z_shifted;

% 初始化一个与 Z_final 大小相同的矩阵，用于存储每个像素的最大值
max_values = Z_shifted;

% 对所有像素，记录下应该修改的位置及其值
rows = size(Z_shifted, 1);
columns = size(Z_shifted, 2);
for r = 1:rows
    for c = 1:columns
        current_value = Z_shifted(r, c);
        
        % 计算左边 3 * 该元素值 个像素的位置
        num_pixels_to_update = min(0.25 * current_value, c - 1);
        if num_pixels_to_update > 0
            left_start = max(c - num_pixels_to_update, 1);
            % 记录需要修改的左边相应像素值
            max_values(r, left_start:c-1) = max(max_values(r, left_start:c-1), current_value);
        end

        % 记录上方的 2 个像素的值
        if r > 1
            max_values(r-1, c) = max(max_values(r-1, c), current_value);
        end
        if r > 2
            max_values(r-2, c) = max(max_values(r-2, c), current_value);
        end

        % 记录下方的 2 个像素的值
        if r < rows
            max_values(r+1, c) = max(max_values(r+1, c), current_value);
        end
        if r < rows - 1
            max_values(r+2, c) = max(max_values(r+2, c), current_value);
        end
    end
end

% 采用记录下来的最大值来更新 Z_final 的值
Z_final = max_values;

% 显示 Z_final 用于验证
figure;
imagesc(Z_final);
colormap('hot'); % 使用 'hot' 颜色映射
colorbar;
xlabel('Column Index');
ylabel('Row Index');
title('Z final with modified variable neighbors for all pixels');