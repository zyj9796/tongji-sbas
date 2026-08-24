% 设置文件名
filename = '/media/mlxg/d/zjc/tianyi2/unwrap/20231007-20231029.diff_flt.mcf.interp.unw';
%filename = '20231029_20240422.cc';
% 设置总行数
lines = 2500;
[data, count] = freadbkB(filename, lines, 'cpxfloat32');

% 显示读取的数据和数量
disp(['Number of elements read: ', num2str(count)]);

% 确认数据是复数
if ~isreal(data)
    % 提取和绘制实部数据
%     data_real = real(data);
%     figure; % 创建一个新的图窗
%     imagesc(data_real); % 使用 imagesc 函数绘制数据
%     colormap('jet'); % 设置颜色映射为 jet
%     colorbar; % 添加颜色条
%     xlabel('Column Index');
%     ylabel('Row Index');
%     title('Real Part of Complex Data');
% 
%     % 提取和绘制虚部数据
%     data_imaginary = imag(data);
%     figure; % 创建一个新的图窗
%     imagesc(data_imaginary); % 使用 imagesc 函数绘制数据
%     colormap('jet'); % 设置颜色映射为 jet
%     colorbar; % 添加颜色条
%     xlabel('Column Index');
%     ylabel('Row Index');
%     title('Imaginary Part of Complex Data');
% 
%     %计算和绘制幅度
%     data_magnitude = abs(data);
%     figure; % 创建一个新的图窗
%     imagesc(data_magnitude); % 使用 imagesc 函数绘制数据
%     colormap('jet'); % 设置颜色映射为 jet
%     colorbar; % 添加颜色条
%     xlabel('Column Index');
%     ylabel('Row Index');
%     title('Magnitude of Complex Data');

    % 计算和绘制相位
  phases = angle(data);

figure; % 创建一个新的图窗
set(gcf, 'Position', [100, 100, 800, 600]); % 设置图窗的位置和大小

imagesc(phases); % 使用 imagesc 函数绘制数据
colormap('jet'); % 设置颜色映射为 jet
colorbar; % 添加颜色条
xlabel('Column Index');
ylabel('Row Index');
title('Phase of Complex Data');
else
    disp('The data does not contain complex values.');
end
