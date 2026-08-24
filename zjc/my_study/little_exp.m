% 简易版本 - 直接处理指定区域的相位数据
% 读取phase数据的相关代码
inft = '/media/mlxg/d/zjc/tianyi2/INTF/20231029-20231120.diff_flt'; % 干涉图文件路径

% 定义处理区域参数
startRow = 3400;    % 起始行
startCol = 6200;    % 起始列
blockRows = 400;    % 区域行数
blockCols = 600;    % 区域列数

% 读取整个图像
lines = 7000; % 计算需要读取的总行数
image = freadbkB(inft, lines, 'cpxfloat32'); % 读取复数浮点类型数据

% 提取指定区域的相位数据
phaseBlock = angle(image(startRow:startRow+blockRows-1, startCol:startCol+blockCols-1));

% 创建对应的掩膜（假设整个区域都是有效数据）
maskBlock = ones(blockRows, blockCols);

% 初始化结果结构
island = struct();
island.Matrix = maskBlock;
island.TopLeftCorner = [startRow, startCol];
island.Size = [blockRows, blockCols];
island.Value = 1;
island.PhaseMatrix = phaseBlock;
island.ResidualPhase = zeros(blockRows, blockCols);
island.UnWarpedPhase = zeros(blockRows, blockCols);
island.HeightMatrix = zeros(blockRows, blockCols);
island.Height = 0;
island.HeightMatrix2 = zeros(blockRows, blockCols);

disp(['Processing phase block at [', num2str(startRow), ',', num2str(startCol), ']']);
disp(['Block size: ', num2str(blockRows), 'x', num2str(blockCols)]);

% 使用你的unwrap_phase_matrix_without_parallel函数进行相位解缠
try
    % 调用你的解缠函数
    [unwrappedPhase, residual] = unwrap_phase_matrix(phaseBlock);
    
    % 存储解缠结果
    island.UnWarpedPhase = unwrappedPhase;
    island.ResidualPhase = residual;
    
    % 相位到高程转换（使用你的phaseToElevation函数）
    heightMatrix = phaseToElevation(unwrappedPhase, island.TopLeftCorner);
    island.HeightMatrix = heightMatrix;
    
    % 计算高程范围
    height_range = max(heightMatrix(:)) - min(heightMatrix(:));
    island.Height = height_range;
    
    % 创建高程矩阵2
    heightMatrix2 = zeros(blockRows, blockCols);
    heightMatrix2(maskBlock == 1) = height_range;
    island.HeightMatrix2 = heightMatrix2;
    
    disp('Phase unwrapping completed successfully!');
    disp(['Height range: ', num2str(height_range)]);
    
catch ME
    disp(['Error during processing: ', ME.message]);
    % 备用方案：使用简单的unwrap方法
    disp('Using fallback unwrap method...');
    unwrappedPhase = simple_unwrap_2d(phaseBlock);
    island.UnWarpedPhase = unwrappedPhase;
    island.ResidualPhase = phaseBlock - mod(unwrappedPhase + pi, 2*pi) - pi;
end

% 可视化结果
figure;
subplot(2,3,1);
imagesc(phaseBlock);
title('原始包裹相位');
colorbar;
axis equal;

subplot(2,3,2);
imagesc(island.UnWarpedPhase);
title('解缠后相位');
colorbar;
axis equal;

subplot(2,3,3);
imagesc(island.ResidualPhase);
title('残差相位');
colorbar;
axis equal;

subplot(2,3,4);
imagesc(island.HeightMatrix);
title('高程矩阵');
colorbar;
axis equal;

subplot(2,3,5);
histogram(island.UnWarpedPhase(:), 50);
title('解缠相位分布');
xlabel('相位值');
ylabel('频数');

subplot(2,3,6);
plot(island.HeightMatrix(round(blockRows/2), :));
title('中心行高程剖面');
xlabel('像素位置');
ylabel('高程值');

% 显示关键统计信息
fprintf('\n=== 处理结果统计 ===\n');
fprintf('区域位置: [%d, %d] 到 [%d, %d]\n', startRow, startCol, startRow+blockRows-1, startCol+blockCols-1);
fprintf('原始相位范围: [%.4f, %.4f] rad\n', min(phaseBlock(:)), max(phaseBlock(:)));
fprintf('解缠相位范围: [%.4f, %.4f] rad\n', min(island.UnWarpedPhase(:)), max(island.UnWarpedPhase(:)));
fprintf('高程变化范围: %.4f 米\n', island.Height);

% 备用简单解缠函数（当主函数不可用时使用）
function unwrapped = simple_unwrap_2d(wrapped_phase)
    % 使用行列法进行二维相位解缠[1,2](@ref)
    [rows, cols] = size(wrapped_phase);
    unwrapped = zeros(rows, cols);
    
    % 先对每一行进行解缠
    for i = 1:rows
        unwrapped(i, :) = unwrap(wrapped_phase(i, :));
    end
    
    % 再对每一列进行解缠[8](@ref)
    for j = 1:cols
        unwrapped(:, j) = unwrap(unwrapped(:, j));
    end
end