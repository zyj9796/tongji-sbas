% 创建图形窗口，并设置大小
figure('Position', [100, 100, 800, 600]); % 800x600 像素的图形窗口
% 显示 data_rmli 图像
imagesc(phases); % 使用 imagesc 函数绘制数据
colormap('jet'); % 背景使用灰色调，以突出叠加的多彩掩膜
%colorbar; % 添加颜色条
% 保持添加新的图层
hold on;
% 使用 jet colormap 生成彩色图像
jetColors = jet(256); % 生成 jet colormap
maskNormalized = (mask_real - min(mask_real(:))) / (max(mask_real(:)) - min(mask_real(:)));
maskNormalized = round(maskNormalized * 255) + 1; % 将数据值映射到 [1, 256]
coloredImage = ind2rgb(maskNormalized, jetColors);


% 显示图像
img_mask = imshow(coloredImage);

% 设置透明度，mask_real 为 0 的地方透明
set(img_mask, 'AlphaData', mask_real > 0);

% 显示坐标轴和网格
axis on;
axis image;
grid on;

% 设置图像标签
xlabel('Column Index');
ylabel('Row Index');
title('Building islands');
hold off;