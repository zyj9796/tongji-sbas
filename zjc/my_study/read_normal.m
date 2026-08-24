% 设置文件名
%filename = '/media/mlxg/d/zjc/my_study/数据/DEM/20231007.dem';
%filename = '/media/mlxg/d/zjc/tianyi2/unwrap/20231007-20231029.diff_flt.mcf.interp.unw';
%filename = '/media/mlxg/d/zjc/tianyi2/unwrap/20231007-20231029.diff_flt.mcf.unw';
%filename = '/media/mlxg/d/zjc/tianyi2/dem/tianjin_DEM.dem';
filename = '/media/mlxg/d/zjc/my_study/数据/DEM/20231007.dem';
% 设置总行数
lines =7000;
%lines=7200;%dem
% 是否执行对数操作
applyLog = false;

% 设置读取的数据范围

% 调用 freadbkB 函数读取指定范围的数据
[data, count] = freadbkB(filename, lines, 'float32');
%data=islands(2).HeightMatrix;
% 显示读取的数据和数量
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

