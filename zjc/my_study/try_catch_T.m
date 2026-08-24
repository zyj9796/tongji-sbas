%% 改进后的版本
% 输入验证
if ~exist('data2','var') || ~ismatrix(data2)
    error('输入必须为矩阵');
end

[rows, cols] = size(data2);
if rows == 1  % 处理行向量情况
    signal = data2;
else
    mid_row = round(rows/2);
    signal = data2(mid_row, :); 
end

%% 参数设置i
window_size = min(30, floor(cols/2)); % 自适应窗口大小
step_size = 1;          
period_thresh = 0.5;   
min_region_length = 10; 
smooth_sigma = 2;       

%% 第一阶段：滑动窗口分析
prob_map = zeros(1, cols);
for pos = 1:step_size:(cols - window_size + 1)
    current_segment = signal(pos:pos+window_size-1);
    
    % 改进的自相关分析
    [acf, lags] = xcorr(current_segment - mean(current_segment), 'coeff');
    acf = acf(lags >= 0);
    [peaks, locs] = findpeaks(acf, 'MinPeakHeight', 0.2);
    
    % 改进的频谱分析
    nfft = 2^nextpow2(window_size);
    [pxx, f] = periodogram(current_segment, [], nfft, 1);
    [~, max_idx] = max(pxx);
    spec_score = min(kurtosis(pxx), 10) * (pxx(max_idx)/mean(pxx)); % 限制峰度最大值
    
    % 综合评分
    if length(peaks) >= 2
        acf_score = min(peaks(2)/peaks(1), 1); % 限制最大比值
    else
        acf_score = 0;
    end
    total_score = acf_score * spec_score;
    
    % 更新概率图
    weights = fspecial('gaussian', [1 window_size], window_size/4);
    prob_map(pos:pos+window_size-1) = prob_map(pos:pos+window_size-1) + total_score * weights;
end

% 安全标准化
if max(prob_map) > min(prob_map)
    prob_map = (prob_map - min(prob_map)) / (max(prob_map) - min(prob_map));
else
    prob_map(:) = 0.5; % 所有值相同的情况
end