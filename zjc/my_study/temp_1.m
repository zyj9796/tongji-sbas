% SAR相位突变检测 - 多横线梯度分析
% 输入：data2 - SAR相位矩阵
% 输出：5个检测位置的突变分析图
data2=islands(16).PhaseMatrix
%% 参数设置
rows = size(data2, 1);
cols = size(data2, 2);
split_lines = round(linspace(1, rows, 6)); % 六等分边界
detect_lines = split_lines(2:6);           % 获取5条检测线位置

%% 多线循环检测
for line_idx = 1:5
    current_row = detect_lines(line_idx);  % 当前检测行号
    phase_line = data2(current_row, :);    % 提取当前行相位
    
    % 梯度计算与分析
    grad = gradient(phase_line);
    abs_grad = abs(grad);
    threshold = 3*std(abs_grad);
    mutant_idx = find(abs_grad > threshold);
    
    % 可视化设置
    fig = figure('Position', [100 100 1200 800],...
                'Name', ['Phase Analysis - Line ', num2str(current_row)],...
                'NumberTitle', 'off');
    
    % 相位曲线
    subplot(3,1,1)
    plot(phase_line, 'b', 'LineWidth', 1.2)
    title(sprintf('横线相位 (行号: %d/%d)', current_row, rows))
    xlabel('水平位置'), ylabel('相位值 (rad)')
    grid on, xlim([1 cols])
    
    % 梯度分析
    subplot(3,1,2)
    plot(grad, 'g', 'LineWidth', 1.2)
    hold on
    plot([1 cols], [threshold threshold], '--r')
    plot([1 cols], [-threshold -threshold], '--r')
    title(sprintf('相位梯度 (阈值: %.2f)', threshold))
    xlabel('水平位置'), ylabel('梯度值')
    legend('梯度','阈值'), grid on, xlim([1 cols])
    
    % 突变点标记
    subplot(3,1,3)
    plot(phase_line, 'b', 'LineWidth', 1.2)
    hold on
    scatter(mutant_idx, phase_line(mutant_idx), 40, 'r', 'filled')
    title(sprintf('检测到 %d 个突变点', length(mutant_idx)))
    xlabel('水平位置'), ylabel('相位值 (rad)')
    legend('原始相位','突变点'), grid on, xlim([1 cols])
    
    % 结果输出
    fprintf('\n检测线 %d/%d (行号 %d):\n', line_idx, 5, current_row);
    fprintf('突变点数量: %d\n', length(mutant_idx));
    fprintf('最大梯度值: %.2f\n', max(abs_grad));
end

%% 辅助说明
disp('分析完成！建议：')
disp('1. 各图窗可通过Figure窗口切换查看')
disp('2. 突变点位置数据保存在工作区的mutant_idx变量中')
disp('3. 当前阈值系数为3σ，可通过修改threshold参数调整灵敏度')