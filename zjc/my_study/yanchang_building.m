
data_geo_extend = data_geo;

% 对每个值大于 40 的像素，设置其左边 3 * 该元素值 个像素，上下 2 个像素为该值
rows = size(data_geo, 1);
columns = size(data_geo, 2);
for r = 1:rows
    for c = 1:columns
        if Z_shifted(r, c) > 40
            current_value = data_geo(r, c);
            % 计算左边 3 * 该元素值 个像素的位置
            num_pixels_to_update = min(0.25 * current_value, c - 1);
            if num_pixels_to_update > 0
                left_start = max(c - num_pixels_to_update, 1);
                % 将左边相应的像素的值设置为当前值
                data_geo_extend(r, left_start:c-1) = current_value;
            end   
        end
    end
end

% 显示 Z_final 用于验证
figure;
imagesc(data_geo_extend);
colormap('hot'); % 使用 'hot' 颜色映射
colorbar;
xlabel('Column Index');
ylabel('Row Index');
title('Z final with values greater than 40 and modified variable left neighbors, up and down neighbors');