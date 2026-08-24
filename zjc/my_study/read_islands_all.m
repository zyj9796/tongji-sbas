% Islands可视化脚本 - 在底图上绘制红色矩形框并显示编号和Heights属性
% 设置工作参数
filename_bmp = '/media/mlxg/d/zjc/tianyi2/INTF/ave.bmp';

% 第一步：检查必要变量是否存在
if ~exist('mask_real', 'var')
    error('mask_real变量未定义，请先加载掩膜数据');
end

if ~exist('islands', 'var')
    error('islands变量未定义，请先运行多干涉对处理程序');
end

numObjects = length(islands);
disp(['找到 ', num2str(numObjects), ' 个岛屿']);

% 第二步：读取BMP底图数据
disp('正在读取底图数据...');
data_rmli = imread(filename_bmp);
disp(['BMP图像尺寸: ', num2str(size(data_rmli, 1)), ' x ', ...
      num2str(size(data_rmli, 2)), ' x ', num2str(size(data_rmli, 3))]);

% 第三步：创建主可视化图形
figure('Position', [100, 100, 1200, 800], 'Name', 'Islands分布图 - 显示Heights属性');

% 显示底图
if size(data_rmli, 3) == 3
    imshow(data_rmli);
else
    imshow(data_rmli, []);
    colormap('gray');
end
hold on;

% 设置绘制参数
rectangleLineWidth = 2;
rectangleEdgeColor = 'r';
textFontSize = 10;

% 第四步：绘制每个岛屿的矩形框和Heights信息
disp('开始绘制岛屿矩形框和Heights信息...');

for i = 1:numObjects
    % 获取岛屿基本信息
    topLeft = islands(i).TopLeftCorner;
    islandSize = islands(i).Size;
    islandHeight = islands(i).Heights;  % 直接获取Heights属性
    
    % 计算矩形位置 [x, y, width, height]
    % 注意：MATLAB中x对应列坐标，y对应行坐标
    x = topLeft(2);  % 列坐标
    y = topLeft(1);  % 行坐标
    width = islandSize(2);
    height = islandSize(1);
    
    % 绘制红色矩形框[1,5](@ref)
    rectangle('Position', [x, y, width, height], ...
              'EdgeColor', rectangleEdgeColor, ...
              'LineWidth', rectangleLineWidth, ...
              'LineStyle', '-');
    
    % 在左上角显示岛屿编号
    text(x, y - 15, ['ID: ', num2str(i)], ...
         'Color', 'black', ...
         'FontSize', textFontSize, ...
         'FontWeight', 'bold', ...
         'BackgroundColor', 'yellow', ...
         'Margin', 1, ...
         'HorizontalAlignment', 'left');
    
    % 在右上角显示Heights属性值
    if ~isnan(islandHeight) && islandHeight > 0
        height_text = sprintf('H: %.1f', islandHeight);
    else
        height_text = 'H: N/A';
    end
    
    text(x + width, y - 15, height_text, ...
         'Color', 'black', ...
         'FontSize', textFontSize, ...
         'FontWeight', 'bold', ...
         'BackgroundColor', 'cyan', ...
         'Margin', 1, ...
         'HorizontalAlignment', 'right');
    
    % 在矩形框中心显示岛屿数值（楼层数）
    text(x + width/2, y + height/2, num2str(islands(i).Value), ...
         'Color', 'white', ...
         'FontSize', textFontSize + 2, ...
         'FontWeight', 'bold', ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'middle', ...
         'BackgroundColor', 'red');
end

% 第五步：添加图表装饰和标签
title('Islands分布图 - 红色矩形框标注（显示Heights属性）', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('列坐标 (像素)');
ylabel('行坐标 (像素)');

% 添加图例
legend('Islands矩形框', 'Location', 'northeastoutside');

% 添加网格
grid on;
grid minor;

% 设置坐标轴范围
axis([1, size(data_rmli, 2), 1, size(data_rmli, 1)]);
axis equal;

hold off;

% 第六步：显示Heights属性的统计信息
disp('=== Heights属性统计信息 ===');
heights = [islands.Heights];
valid_heights = heights(~isnan(heights) & heights > 0);

if ~isempty(valid_heights)
    fprintf('Heights属性范围: %.2f - %.2f\n', min(valid_heights), max(valid_heights));
    fprintf('Heights属性平均值: %.2f\n', mean(valid_heights));
    fprintf('Heights属性标准差: %.2f\n', std(valid_heights));
else
    fprintf('无有效Heights数据\n');
end

% 第七步：保存结果图像
disp('正在保存结果图像...');
saveas(gcf, 'islands_heights_visualization.png', 'png');
disp('可视化图像已保存: islands_heights_visualization.png');

% 第八步：显示详细的Heights信息
fprintf('\n=== 各岛屿Heights属性详情 ===\n');
fprintf('岛屿ID\tHeights属性值\n');
for i = 1:numObjects
    if ~isnan(islands(i).Heights) && islands(i).Heights > 0
        fprintf('%d\t%.2f\n', i, islands(i).Heights);
    else
        fprintf('%d\tN/A\n', i);
    end
end

disp('岛屿Heights属性可视化完成！');