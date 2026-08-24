% 读取phase数据的相关代码
% inft = 'H:\tianyi2\INTF\20240422-20240503.diff'; % 干涉图文件路径
inft = '/media/mlxg/d/zjc/tianyi2/INTF/20231007-20231029.diff_flt'; % 干涉图文件路径
image = freadbkB(inft, lines, 'cpxfloat32'); % 读取复数浮点类型数据
phases = angle(image); % 获取相位信息
% 使用 bwlabel 找到连通分量
[labelMatrix, numObjects] = bwlabel(mask_real, 4); % 4 或 8 连通性
% 初始化结构体数组
islands = struct('Matrix', {}, 'TopLeftCorner', {}, 'Size', {}, 'Value', {}, 'PhaseMatrix', {}, 'UnWarpedPhase', {}, 'ResidualPhase', {}, 'HeightMatrix', {}, 'Height', {},'HeightMatrix2', {});

for i = 1:numObjects
    % 找到当前连通分量的像素
    pixelIdx = find(labelMatrix == i);
    
    % 将线性索引转换为 [row, col] 索引
    [rows, cols] = ind2sub(size(mask_real), pixelIdx);
    
    % 找到这些索引的边界
    minRow = min(rows);
    maxRow = max(rows);
    minCol = min(cols);
    maxCol = max(cols);
    
    % 提取孤岛的子矩阵
    islandMatrix = mask_real(minRow:maxRow, minCol:maxCol);
    
    % 提取相应的 phase 子矩阵
    phaseMatrix = phases(minRow:maxRow, minCol:maxCol);
    
    % 孤岛矩阵的左上角坐标
    topLeftCorner = [minRow, minCol];
    
    % 孤岛矩阵的大小
    sizeOfIsland = size(islandMatrix);

    % 创建一个与孤岛矩阵尺寸相同的零矩阵
    unwarpedPhase = zeros(sizeOfIsland);
    
    % 创建另一个与孤岛矩阵尺寸相同的零矩阵
    heightMatrix = zeros(sizeOfIsland);
    
    % 创建另一个与孤岛矩阵尺寸相同的零矩阵
    residualMatrix = zeros(sizeOfIsland);
    
    % 获取孤岛的数值（使用众数）
    islandValue = mode(mask_real(pixelIdx));
  
    % 遍历并修改 phaseMatrix，当 islandMatrix 的像素值不等于 islandValue 时，将 phaseMatrix 的像素值置为 0
    for r = 1:size(phaseMatrix, 1)
        for c = 1:size(phaseMatrix, 2)
            if islandMatrix(r, c) ~= islandValue
                phaseMatrix(r, c) = 0;
            end
        end
    end

    % 将提取的信息存入结构体中
    islands(i).Matrix = islandMatrix;
    islands(i).TopLeftCorner = topLeftCorner;
    islands(i).Size = sizeOfIsland;
    islands(i).Value = islandValue;
    islands(i).PhaseMatrix = phaseMatrix;
    islands(i).ResidualPhase = residualMatrix; % WarpedPhase
    islands(i).UnWarpedPhase = unwarpedPhase; % WarpedPhase
    islands(i).HeightMatrix = heightMatrix; % HeightMatrix
end

% 输出结构体数组以查看结果
disp("mask divide finished!");

% 遍历每个 island
for i = 1:numel(islands)
    % 提取当前岛的 PhaseMatrix
    phaseMatrix = islands(i).PhaseMatrix;
    disp(['Processing island number ' num2str(i)]);
    % 调用 unwrap_phase_matrix 函数进行处理
    [unwrappedPhase, residual] = unwrap_phase_matrix_without_parallel(phaseMatrix);
    % 将结果存入对应的结构字段中
    islands(i).UnWarpedPhase = unwrappedPhase;
    islands(i).ResidualPhase = residual;
    heightMatrix = phaseToElevation(unwrappedPhase,islands(i).TopLeftCorner);
    %islands(i).UnWarpedPhase = fliplr(unwrappedPhase);
    islands(i).HeightMatrix = heightMatrix;
    height = max(heightMatrix(:)) - min(heightMatrix(:)); % 找到最大值
    islands(i).Height = height;
    heightMatrix2 = zeros(islands(i).Size);         % 复制 heightMatrix
    heightMatrix2(islands(i).Matrix == islands(i).Value) = height; % 将所有非零元素设置为 height
    %heightMatrix2 = unwrappedPhase; % 将所有非零元素设置为 height
    islands(i).HeightMatrix2 = heightMatrix2;
end

% 打印处理完成的消息
disp('Phase unwrapping completed for all islands!');