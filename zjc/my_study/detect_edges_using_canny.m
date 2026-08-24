function edges = detect_edges_using_canny(data_phase_log)
    % 输入的data_phase_log应该是一个二维矩阵
    
    % Step 1: 归一化输入数据到0-255
    data_min = min(data_phase_log(:));
    data_max = max(data_phase_log(:));
    data_norm = (data_phase_log - data_min) / (data_max - data_min) * 255;
    
    % Step 2: 将数据类型转换为uint8
    data_uint8 = uint8(data_norm);
    
    % Step 3: 高斯平滑处理
    data_smoothed = imgaussfilt(data_uint8, 0.1);  % sigma = 1 可以根据需要调整
    
    % Step 4: 应用Canny边缘检测算法
    threshold_low = 0.4;  % 调整低阈值
    threshold_high = 0.8; % 调整高阈值
    edges = edge(data_smoothed, 'Canny', [threshold_low threshold_high]);
    
    % 可视化结果
    figure;
    subplot(1, 3, 1);
    imagesc(data_uint8);
    colormap(gray);
    axis image;
    title('Normalized Input Data');
    
    subplot(1, 3, 2);
    imagesc(data_smoothed);
    colormap(gray);
    axis image;
    title('Smoothed Data');
    
    subplot(1, 3, 3);
    imagesc(edges);
    colormap(gray);
    axis image;
    title('Detected Edges (Canny)');
end