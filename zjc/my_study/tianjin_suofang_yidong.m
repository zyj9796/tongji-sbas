% 假设 Z 已经存在并且是 250x250 的矩阵
% 例如:
% Z = randi([0, 255], 250, 250); % 随机生成一个250x250的矩阵用于测试

% 左平移 20 像素处理
shift_left_amount = 4;
Z_shifted_left = -ones(size(Z)); % 创建填充 -1 的矩阵
Z_shifted_left(:, 1:end-shift_left_amount) = Z(:, shift_left_amount+1:end); % 将原 Z 的内容左移

% 下平移 5 像素处理
shift_down_amount = 5;
Z_shifted = -ones(size(Z)); % 创建填充 -1 的矩阵
Z_shifted(shift_down_amount+1:end, :) = Z_shifted_left(1:end-shift_down_amount, :); % 将已左移后的 Z 向下移动

% 显示 Z_shifted 用于验证
figure;
imagesc(Z_shifted);
colormap('hot'); % 使用 'hot' 颜色映射
colorbar;
xlabel('Column Index');
ylabel('Row Index');
title('Z shifted left by 20 pixels and down by 5 pixels with -1 padding');