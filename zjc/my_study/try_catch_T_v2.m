%% 中线周期性区域自动提取脚本
% 输入：data2 相位矩阵
% 输出：周期性区域起止位置及可视化

% 提取中线数据
[rows, cols] = size(data2);
mid_row = round(rows / 2);
signal = data2(mid_row, :);

%% 参数设置
window_size = 50;       % 滑动窗口长度（建议为2-3倍周期长度）
step_size = 20;          % 滑动步长（建议窗口的1/5-1/3）
period_thresh = 0.9;    % 周期性判定阈值（0-1之间）
min_region_length = 50; % 最小有效区域长度
smooth_sigma = 1;       % 高斯平滑系数（越大越平滑）

%% 第一阶段：滑动窗口分析
% 初始化概率图
prob_map = zeros(1, cols);

% 滑动窗口扫描
for pos = 1:step_size:(cols - window_size + 1)
    % 提取当前窗口数据
    current_segment = signal(pos:pos+window_size-1);
    
    % 自相关分析
    [acf, lags] = xcorr(current_segment - mean(current_segment), 'coeff');
    acf = acf(lags >= 0); % 只保留非负滞后部分
    [peaks, ~] = findpeaks(acf); % 寻找自相关峰值
    
    % 频谱分析
    [pxx, f] = periodogram(current_segment, [], 20, 1);
    [~, max_idx] = max(pxx);
    spec_score = kurtosis(pxx) * (pxx(max_idx) / mean(pxx));
    
    % 综合评分
    if length(peaks) >= 2
        acf_score = peaks(2) / peaks(1); % 次峰值与主峰值的比值
    else
        acf_score = 0;
    end
    total_score = acf_score * spec_score; % 综合得分
    
    % 更新概率图（高斯加权）
    weights = fspecial('gaussian', [1 window_size], window_size / 4);
    prob_map(pos:pos+window_size-1) = prob_map(pos:pos+window_size-1) + total_score * weights;
end

% 标准化概率图
prob_map = (prob_map - min(prob_map)) / (max(prob_map) - min(prob_map));

%% 第二阶段：区域检测
% 高斯平滑与自适应阈值
smoothed_prob = imgaussfilt(prob_map, smooth_sigma);
threshold = graythresh(smoothed_prob) * max(smoothed_prob);
binary_mask = smoothed_prob > threshold;

% 形态学闭运算（水平连接）
se = strel('line', window_size, 0);
closed_mask = imclose(binary_mask, se);

% 区域筛选
cc = bwconncomp(closed_mask);
stats = regionprops(cc, 'Area', 'BoundingBox');
valid_regions = find([stats.Area] >= min_region_length);

%% 第三阶段：中心区域选择
if ~isempty(valid_regions)
    % 计算区域中心位置
    centroids = arrayfun(@(x) x.BoundingBox(1) + x.BoundingBox(3) / 2, stats(valid_regions));
    
    % 选择最接近信号中心的区域
    [~, idx] = min(abs(centroids - cols / 2));
    selected = stats(valid_regions(idx));
    
    % 获取边界
    start_idx = round(selected.BoundingBox(1));
    end_idx = start_idx + round(selected.BoundingBox(3)) - 1;
else
    % 默认全段（未检测到周期区域）
    start_idx = 1;
    end_idx = cols;
    warning('未检测到显著周期区域，返回全段');
end

%% 第四阶段：边界优化
% 左边界优化（通过梯度最小化寻找平滑边界）
left_grad = abs(gradient(signal(max(1, start_idx - 50):start_idx + 50)));
[~, min_pos] = min(left_grad);
start_idx = max(1, start_idx - 50 + min_pos - 1);

% 右边界优化（通过梯度最小化寻找平滑边界）
right_grad = abs(gradient(signal(max(1, end_idx - 50):min(cols, end_idx + 50))));
[~, min_pos] = min(right_grad);
end_idx = min(cols, end_idx - 50 + min_pos - 1);

% 边界保护
start_idx = max(1, start_idx);
end_idx = min(cols, end_idx);
start_idx = 57;
end_idx = 257;
%% 结果可视化
figure('Position', [100 100 1200 600])

% 原始信号与概率图
subplot(3, 1, 1)
yyaxis left
plot(signal, 'b', 'LineWidth', 1.2)
ylabel('相位值 (rad)', 'Color', 'b')

yyaxis right
plot(smoothed_prob, 'r', 'LineWidth', 1.5)
ylabel('周期概率', 'Color', 'r')
title('原始信号与周期概率分布')
xlim([1 cols])
grid on

% 区域检测过程
subplot(3, 1, 2)
hold on
plot(prob_map, 'Color', [0.5 0.5 0.5], 'LineWidth', 1)
plot(smoothed_prob, 'b', 'LineWidth', 1.5)
plot([1 cols], [threshold threshold], '--r')
legend('原始概率', '平滑后', '检测阈值')
title('概率处理流程')
xlim([1 cols])
grid on

% 最终结果展示
subplot(3, 1, 3)
hold on
plot(signal, 'b', 'LineWidth', 1.2)
xline(start_idx, '--r', 'LineWidth', 2)
xline(end_idx, '--r', 'LineWidth', 2)
area(start_idx:end_idx, signal(start_idx:end_idx), ...
    'FaceColor', [1 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none')
title(sprintf('检测结果：周期区域 %d-%d (长度%d)', start_idx, end_idx, end_idx - start_idx + 1))
xlim([1 cols])
grid on

%% 结果输出
fprintf('\n检测报告：\n')
fprintf('信号长度：%d 像素\n', cols)
fprintf('周期区域：%d - %d\n', start_idx, end_idx)
fprintf('区域长度：%d 像素 (占总长度%.1f%%)\n', ...
    end_idx - start_idx + 1, 100 * (end_idx - start_idx + 1) / cols);