% 一维多横线周期性检测（基于data2相位矩阵）
% 输入：data2 = islands(7).PhaseMatrix
% 输出：各检测线的周期性区域标记及可视化

%% 参数设置
num_lines = 5;                   % 检测线数量
window_size = 80;                % 滑动窗口长度（建议≥3个周期）
step_size = 15;                   % 滑动步长（建议窗口的1/5-1/3）
period_thresh = 0.65;            % 自相关阈值（0-1）
min_segment_length = 50;         % 最小有效周期段长度（像素）

%% 六等分横线定位
[rows, cols] = size(data2);
split_pos = round(linspace(1, rows, 6));  % 六等分位置
detect_lines = split_pos(2:6);            % 获取5条检测线

%% 各线独立分析
figure('Position', [100 100 1200 700], 'Name', '一维周期性检测分析')

for line_idx = 1:num_lines
    % 当前检测线数据
    current_line = detect_lines(line_idx);
    phase_line = data2(current_line, :);
    
    %% 阶段1：周期性窗口检测
    periodic_mask = false(1, cols);  % 初始化掩膜
    
    % 滑动窗口分析
    for pos = 1:step_size:(cols-window_size)
        segment = phase_line(pos:pos+window_size-1);
        
        % 自相关分析
        [acf, lags] = xcorr(segment-mean(segment), 'coeff');
        acf = acf(lags>=0);
        [peaks, locs] = findpeaks(acf);
        
        % 周期性判定
        if length(peaks)>=2
            main_peak = peaks(1);
            side_peak = peaks(2);
            periodicity = side_peak/main_peak;
            
            if periodicity > period_thresh
                periodic_mask(pos:pos+window_size-1) = true;
            end
        end
    end
    
    %% 阶段2：一维形态学优化
    % 水平方向闭运算（连接相邻窗口）
    se = strel('line', window_size/2, 0); % 水平结构元素
    closed_mask = imclose(periodic_mask, se);
    
    % 区域长度过滤
    cc = bwconncomp(closed_mask);
    stats = regionprops(cc, 'Area', 'PixelIdxList');
    for k = 1:cc.NumObjects
        if stats(k).Area < min_segment_length
            closed_mask(stats(k).PixelIdxList) = false;
        end
    end
    
    %% 可视化（每线一个子图）
    subplot(num_lines, 1, line_idx)
    hold on
    
    % 绘制原始相位曲线
    plot(phase_line, 'b', 'LineWidth', 1.2, 'DisplayName', '相位值')
    
    % 标记周期性区域
    [start_idx, end_idx] = find_contiguous(closed_mask);
    for seg = 1:length(start_idx)
        x_range = start_idx(seg):end_idx(seg);
        y_values = phase_line(x_range);
        area(x_range, y_values,...
            'FaceColor', [0.8 0.2 0.2], 'FaceAlpha', 0.3,...
            'EdgeColor', 'none', 'DisplayName', '周期区域');
    end
    
    % 图例与标注
    title(sprintf('检测线 %d (行号: %d)', line_idx, current_line))
    xlabel('水平位置'), ylabel('相位值 (rad)')
    xlim([1 cols])
    grid on
    if line_idx == 1
        legend('show', 'Location', 'best')
    end
end

%% 辅助函数：查找连续区域边界
function [start_idx, end_idx] = find_contiguous(mask)
    diff_mask = diff([false mask false]);
    start_idx = find(diff_mask == 1);
    end_idx = find(diff_mask == -1) - 1;
end