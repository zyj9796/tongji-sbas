% 读取phase数据的相关代码
% inft = 'H:\tianyi2\INTF\20240422-20240503.diff'; % 干涉图文件路径
inft = 'H:\tianyi2\INTF\20240503-20240514.diff_flt'; % 干涉图文件路径
image = freadbkB(inft, lines, 'cpxfloat32'); % 读取复数浮点类型数据
phases = angle(image); % 获取相位信息
phases = phases(:, 1:6600);

% 使用自定义函数进行多值连通域标记
[labelMatrix, numObjects] = multiLabel(mask_real, 8); 

% 初始化结构体数组
islands = struct('Matrix', {}, 'TopLeftCorner', {}, 'Size', {}, 'Value', {}, ...
                 'PhaseMatrix', {}, 'UnWarpedPhase', {}, 'ResidualPhase', {}, ...
                 'HeightMatrix', {}, 'Height', {},'HeightMatrix2', {});

for i = 1:numObjects
    % 获取当前区域像素坐标
    [rows, cols] = find(labelMatrix == i);
    
    % 计算边界框
    minRow = min(rows);
    maxRow = max(rows);
    minCol = min(cols);
    maxCol = max(cols);
    
    % 提取子矩阵（扩展边界外部分用0填充）
    islandMatrix = mask_real(minRow:maxRow, minCol:maxCol);
    phaseMatrix = phases(minRow:maxRow, minCol:maxCol);
    
    % 生成掩膜并清理相位数据
    regionMask = (labelMatrix(minRow:maxRow, minCol:maxCol) == i);
    phaseMatrix(~regionMask) = 0; % 向量化操作替换循环
    
    % 获取区域特征值（直接取第一个像素值）
    islandValue = mask_real(rows(1), cols(1));
    
    % 写入结构体
    islands(i).Matrix = islandMatrix;
    islands(i).TopLeftCorner = [minRow, minCol];
    islands(i).Size = size(islandMatrix);
    islands(i).Value = islandValue;
    islands(i).PhaseMatrix = phaseMatrix;
    islands(i).ResidualPhase = zeros(size(islandMatrix)); 
    islands(i).UnWarpedPhase = zeros(size(islandMatrix));
    islands(i).HeightMatrix = zeros(size(islandMatrix));
end

disp("Multi-value mask division completed!");

% 遍历每个 island
for i = 1:numel(islands)
    % 提取当前岛的 PhaseMatrix
    phaseMatrix = islands(i).PhaseMatrix;
    disp(['Processing island number ' num2str(i)]);
    % 调用 unwrap_phase_matrix 函数进行处理
    [unwrappedPhase, residual] = unwrap_phase_matrix(phaseMatrix);
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

% 遍历每个 island
for i = 1:numel(islands)
    % 提取当前岛的 PhaseMatrix
    phaseMatrix = islands(i).PhaseMatrix;
    disp(['Processing island number ' num2str(i)]);
    % 调用 unwrap_phase_matrix 函数进行处理
    [unwrappedPhase, residual] = unwrap_phase_matrix(phaseMatrix);
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