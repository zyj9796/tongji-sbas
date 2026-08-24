% 清除所有变量并关闭所有图形窗口
clear;
close all;

% 读取 TIFF 图像文件
imageFileName = 'tianjin.tif';
imageData = imread(imageFileName);

% 获取图像的地理信息（选项）
% info = imfinfo(imageFileName);
% 注意：如果你有地理信息（比如从GeoTIFF文件），可以用imfinfo函数获取并使用

% 假设你知道图像实际显示时长宽的比例，例如：
long_ratio = 0.7; % 假设宽度占比例更大
lat_ratio = 1;  % 假设高度占比例更小

% 显示图像
figure;
imshow(imageData);
title('Tianjin Image');

% 设置坐标轴，以反映实际的长宽比例
axis equal; % 保持像素大小相等
truesize; % 确保图像保持原始尺寸

% 还可以手动调整坐标轴
ax = gca;
ax.DataAspectRatio = [long_ratio lat_ratio 1]; % 设置数据长宽高比