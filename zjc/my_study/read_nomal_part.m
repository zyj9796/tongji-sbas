% 设置文件名
filename = '/media/mlxg/d/zjc/tianyi2/past_process/unwrap/20231109-20231120.diff_flt.mcf.corrected.unw';
% 设置总行数
lines =2000;
applyLog = false;
% 定义处理区域参数
startRow = 1000;    % 起始行
startCol = 1700;    % 起始列
blockRows = 400;    % 区域行数
blockCols = 600;    % 区域列数

% 设置读取的数据范围
% 调用 freadbkB 函数读取指定范围的数据
[data, count] = freadbkB(filename, lines, 'float32');
% 显示读取的数据和数量
data=data(startRow:startRow+blockRows-1, startCol:startCol+blockCols-1);
disp(['Number of elements read: ', num2str(count)]);

% 取对数操作（如果需要）
if applyLog
    data = log10(data + 1e-10); % 加1防止log(0)的情况
end
% 显示数据为图像
figure;
set(gcf, 'Position', [100, 100, 800, 600]);
%data=3*data;
%data=Output.demerror;
imagesc(data); % 或者使用 imshow(dataMatrix, []);
colorbar; % 显示颜色条

xlabel('Column Index');
ylabel('Row Index');
title('wrappedphase');

% 设置颜色图（可选）
%colormap('jet');

