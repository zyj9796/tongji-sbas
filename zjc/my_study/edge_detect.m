% 假设你已经加载或生成了一个data_phase_log矩阵
%edges = detect_edges_using_canny(Z_shifted);
% 假设 Z_shifted 矩阵已经存在
% Z_shifted = <your_Z_shifted_matrix>;

% 这里假设 Z_shifted 已经被定义
% 示例：Z_shifted = magic(10);  % 这只是一个示例矩阵，用你的矩阵替换它

% 将 Z_shifted 矩阵标准化到 [0, 1] 范围，如果它包含浮点数或大于1的数值
Z_shifted = mat2gray(Z_shifted);

% 使用 Sobel 算子进行边缘检测
edges_sobel = edge(Z_shifted, 'Sobel');
figure;
imshow(edges_sobel);
title('Sobel 边缘检测');

% 使用 Canny 算子进行边缘检测
edges_canny = edge(Z_shifted, 'Canny');
figure;
imshow(edges_canny);
title('Canny 边缘检测');

% 使用 Prewitt 算子进行边缘检测
edges_prewitt = edge(Z_shifted, 'Prewitt');
figure;
imshow(edges_prewitt);
title('Prewitt 边缘检测');

% 使用 Laplace 算子进行边缘检测
edges_laplace = edge(Z_shifted, 'log'); % log 是 Laplacian of Gaussian
figure;
imshow(edges_laplace);
title('Laplace 边缘检测');