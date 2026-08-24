% 假设 Z 和 data_phase 已经存在并且都是 250x250 的矩阵

% 创建图形窗口，并设置大小
figure('Position', [100, 100, 800, 800]); % 800x800 像素的图形窗口

% 显示 data_phase 图像
imagesc(data_phase_log); % 使用 imagesc 函数绘制数据
colormap('jet'); % 设置颜色映射为 jet
colorbar; % 添加颜色条

% 保持添加新的图层
hold on;

% 使用红色显示 Z 图像，将 Z 小于 5 的值设为透明
redMask = Z_final >= 40;

% 创建白色图像
whiteImage = cat(3, double(redMask), double(redMask), double(redMask)); % 白色通道

% 创建窗口并显示白色图像
hZ = imshow(whiteImage);

% 设置透明度
set(hZ, 'AlphaData', redMask); % 设置透明度


% 显示坐标轴和网格
axis on;
axis image;
grid on;

% 设置图像标签
xlabel('Column Index');
ylabel('Row Index');
title('Overlay of data_phase (Jet) and Z (Red)');
hold off;