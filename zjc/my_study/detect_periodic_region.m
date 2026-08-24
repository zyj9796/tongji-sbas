function [start_idx, end_idx] = detect_periodic_region(signal, params)
% 检测信号中的周期性区域
% 输入：
%   signal - 输入信号（行向量）
%   params - 参数结构体（可选）
% 输出：
%   start_idx, end_idx - 周期性区域起止位置

%% 默认参数配置（可调整部分）
if nargin < 2
    params = struct();
end

% 滑动窗口参数
params = set_default(params, 'window_size', 30);       % 建议2-3倍预期周期长度
params = set_default(params, 'step_size', 1);          % 滑动步长
params = set_default(params, 'min_window_ratio', 0.1); % 最小窗口/信号长度比

% 周期性检测参数
params = set_default(params, 'acf_thresh', 0.3);      % 自相关峰值阈值
params = set_default(params, 'harmonic_ratio', 0.7);  % 谐波能量比
params = set_default(params, 'min_peaks', 2);         % 最小峰值数量

% 区域处理参数
params = set_default(params, 'smooth_sigma', 2);      % 高斯平滑系数
params = set_default(params, 'min_region_length', 10);% 最小区域长度
params = set_default(params, 'edge_buffer', 50);     % 边界优化缓冲

%% 信号预处理
signal = signal(:)'; % 确保行向量
N = length(signal);

% 自适应调整窗口大小
min_window = max(3, round(N*params.min_window_ratio));
params.window_size = min(params.window_size, N);
params.window_size = max(params.window_size, min_window);

%% 主检测算法（基于谐波分析和自相关）
prob_map = zeros(1, N);
window = hamming(params.window_size)'; % 汉明窗减少频谱泄漏

for pos = 1:params.step_size:(N - params.window_size + 1)
    segment = signal(pos:pos+params.window_size-1);
    segment = segment - mean(segment); % 去除直流分量
    
    % 方法1：自相关分析
    [acf, lags] = xcorr(segment, 'coeff');
    acf = acf(lags >= 0);
    [peaks, locs] = findpeaks(acf, 'MinPeakHeight', params.acf_thresh);
    
    % 方法2：谐波分析
    [pxx, f] = periodogram(segment, window, [], 1);
    [max_pxx, max_idx] = max(pxx(2:end)); % 忽略直流
    fund_freq = f(max_idx+1);
    
    % 计算谐波能量比
    harmonic_bw = round(0.1*length(f)); % 谐波带宽
    harmonic_energy = sum(pxx(max_idx+1-harmonic_bw:max_idx+1+harmonic_bw));
    total_energy = sum(pxx);
    hr_ratio = harmonic_energy/total_energy;
    
    % 综合评分
    if length(peaks) >= params.min_peaks
        period_score = min(peaks(2)/peaks(1) * hr_ratio;
    else
        period_score = 0;
    end
    
    % 更新概率图（高斯加权）
    weights = gausswin(params.window_size)';
    prob_map(pos:pos+params.window_size-1) = prob_map(pos:pos+params.window_size-1) + ...
        period_score * weights;
end

% 标准化概率图
prob_map = (prob_map - min(prob_map)) / (max(prob_map) - min(prob_map) + eps);

%% 区域提取与优化
% 高斯平滑
smoothed_prob = imgaussfilt(prob_map, params.smooth_sigma);

% 自适应阈值
threshold = graythresh(smoothed_prob) * max(smoothed_prob);
binary_mask = smoothed_prob > threshold;

% 形态学处理
se = strel('line', params.window_size, 0);
closed_mask = imclose(binary_mask, se);

% 区域选择
cc = bwconncomp(closed_mask);
stats = regionprops(cc, 'Area', 'BoundingBox');
valid_regions = find([stats.Area] >= params.min_region_length);

if isempty(valid_regions)
    start_idx = 1;
    end_idx = N;
    warning('未检测到显著周期区域');
else
    % 选择最大概率区域
    region_scores = arrayfun(@(x) mean(prob_map(...
        round(x.BoundingBox(1)):round(x.BoundingBox(1)+x.BoundingBox(3)))), ...
        stats(valid_regions));
    [~, best_idx] = max(region_scores);
    selected = stats(valid_regions(best_idx));
    
    % 获取初始边界
    start_idx = max(1, round(selected.BoundingBox(1)));
    end_idx = min(N, start_idx + round(selected.BoundingBox(3)) - 1);
    
    % 边界优化
    opt_len = min(params.edge_buffer, round((end_idx-start_idx)/2));
    if opt_len > 5
        [~, start_idx] = min(abs(gradient(signal(start_idx:start_idx+opt_len))));
        start_idx = start_idx + start_idx - 1;
        
        [~, end_opt] = min(abs(gradient(signal(end_idx-opt_len:end_idx))));
        end_idx = end_idx - opt_len + end_opt - 1;
    end
end

%% 可视化（可选）
if nargout == 0 || exist('params', 'var') && isfield(params, 'visualize') && params.visualize
    figure('Position', [100 100 1200 800])
    
    subplot(4,1,1)
    plot(signal, 'b')
    title('原始信号')
    xlim([1 N])
    
    subplot(4,1,2)
    plot(prob_map, 'r')
    hold on
    plot(smoothed_prob, 'k', 'LineWidth', 1.5)
    plot([1 N], [threshold threshold], '--g')
    title('周期概率分布')
    legend('原始', '平滑', '阈值')
    xlim([1 N])
    
    subplot(4,1,3)
    imagesc(closed_mask)
    title('区域检测结果')
    colormap(gray)
    
    subplot(4,1,4)
    plot(signal, 'b')
    hold on
    xline(start_idx, '--r')
    xline(end_idx, '--r')
    area(start_idx:end_idx, signal(start_idx:end_idx), ...
        'FaceColor', [1 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none')
    title(sprintf('最终结果: %d-%d (长度%d)', start_idx, end_idx, end_idx-start_idx+1))
    xlim([1 N])
end

%% 辅助函数
function params = set_default(params, field, value)
if ~isfield(params, field)
    params.(field) = value;
end