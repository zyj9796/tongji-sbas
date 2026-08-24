% 多对干涉对高程解算主程序（优化版）
% 设置工作参数
intf_dir = '/media/mlxg/d/zjc/tianyi2/INTF/'; % 干涉图文件夹路径
lines = 7000; % 图像行数

% 第一步：彻底抑制所有警告输出
warning('off', 'all');
fprintf('所有警告输出已彻底抑制\n');

% 第二步：获取所有干涉图文件列表
disp('正在扫描干涉图文件...');
file_list = dir(fullfile(intf_dir, '*.diff_flt'));
num_interferograms = length(file_list);
disp(['找到 ', num2str(num_interferograms), ' 个干涉图文件']);

% 检查是否有找到文件
if num_interferograms == 0
    error('在指定目录下未找到任何.diff_flt文件');
end

% 显示文件列表
for i = 1:num_interferograms
    disp(['文件 ', num2str(i), ': ', file_list(i).name]);
end

% 第三步：读取mask_real（假设mask_real已经存在）
% 这里需要确保mask_real已经被正确加载
if ~exist('mask_real', 'var')
    error('mask_real变量未定义，请先加载掩膜数据');
end

% 修改点1：使用基于值一致性的连通分量标记替代bwlabel
[labelMatrix, numObjects] = valueBasedConnectedComponents(mask_real, 4); % 4连通性，但考虑像素值一致性

% 初始化结构体数组 - 修改字段以支持多干涉对
islands = struct('Matrix', {}, 'TopLeftCorner', {}, 'Size', {}, 'Value', {}, ...
                 'PhaseMatrixs', {}, 'UnWarpedPhases', {}, 'ResidualPhases', {}, ...
                 'HeightMatrixs', {}, 'Heights', {});

% 第四步：为每个island预先分配空间并提取基本信息
disp('正在初始化岛屿结构...');
for i = 1:numObjects
    % 找到当前连通分量的像素
    pixelIdx = find(labelMatrix == i);
    
    % 将线性索引转换为[row, col]索引
    [rows, cols] = ind2sub(size(mask_real), pixelIdx);
    
    % 找到这些索引的边界
    minRow = min(rows);
    maxRow = max(rows);
    minCol = min(cols);
    maxCol = max(cols);
    
    % 提取孤岛的子矩阵
    islandMatrix = mask_real(minRow:maxRow, minCol:maxCol);
    
    % 孤岛矩阵的左上角坐标
    topLeftCorner = [minRow, minCol];
    
    % 孤岛矩阵的大小
    sizeOfIsland = size(islandMatrix);
    
    % 获取孤岛的数值（使用众数）
    islandValue = mode(mask_real(pixelIdx));
    
    % 为多干涉对预分配空间
    % PhaseMatrixs: m×n×N 三维矩阵
    phaseMatrixs = zeros([sizeOfIsland, num_interferograms]);
    
    % 同样为解缠结果和残差预分配空间
    unwarpedPhases = zeros([sizeOfIsland, num_interferograms]);
    residualPhases = zeros([sizeOfIsland, num_interferograms]);
    heightMatrixs = zeros([sizeOfIsland, num_interferograms]);
    heights = zeros(1, num_interferograms);
    
    % 将基本信息存入结构体
    islands(i).Matrix = islandMatrix;
    islands(i).TopLeftCorner = topLeftCorner;
    islands(i).Size = sizeOfIsland;
    islands(i).Value = islandValue;
    islands(i).PhaseMatrixs = phaseMatrixs;
    islands(i).UnWarpedPhases = unwarpedPhases;
    islands(i).ResidualPhases = residualPhases;
    islands(i).HeightMatrixs = heightMatrixs;
    islands(i).Heights = heights;
end

disp(['岛屿结构初始化完成，共 ', num2str(numObjects), ' 个岛屿']);

% 第五步：读取所有干涉图数据并填充PhaseMatrixs
disp('开始读取干涉图数据...');
for file_idx = 1:num_interferograms
    current_file = fullfile(intf_dir, file_list(file_idx).name);
    disp(['处理干涉图 ', num2str(file_idx), '/', num2str(num_interferograms), ': ', file_list(file_idx).name]);
    
    try
        % 读取复数干涉图数据
        image_data = freadbkB(current_file, lines, 'cpxfloat32');
        phases = angle(image_data); % 获取相位信息
        
        % 为每个island提取对应的相位区域
        for i = 1:numObjects
            topLeft = islands(i).TopLeftCorner;
            islandSize = islands(i).Size;
            islandMatrix = islands(i).Matrix;
            islandValue = islands(i).Value;
            
            % 提取相应的phase子矩阵
            minRow = topLeft(1);
            minCol = topLeft(2);
            maxRow = minRow + islandSize(1) - 1;
            maxCol = minCol + islandSize(2) - 1;
            
            phasePatch = phases(minRow:maxRow, minCol:maxCol);
            
            % 应用掩膜：只有当islandMatrix的值等于islandValue时保留相位值
            phasePatch(islandMatrix ~= islandValue) = 0;
            
            % 将处理后的相位数据存入PhaseMatrixs的对应层
            islands(i).PhaseMatrixs(:, :, file_idx) = phasePatch;
        end
        
    catch ME
        fprintf('处理文件失败: %s, 错误: %s\n', current_file, ME.message);
        % 如果某个文件读取失败，将该层的相位数据设为NaN
        for i = 1:numObjects
            islands(i).PhaseMatrixs(:, :, file_idx) = NaN;
        end
    end
end

disp('所有干涉图数据读取完成！');

% 第六步：启动并行池并配置并行环境
disp('启动并行池用于相位解缠...');
try
    % 检查是否已有并行池
    pool = gcp('nocreate');
    if isempty(pool)
        % 获取可用核心数并启动并行池
        %numCores = feature('numcores');
        % 限制最大工作进程数避免内存问题
        %desiredWorkers = min(numCores, 128);
        parpool(128);
        fprintf('并行池已启动，使用 %d 个工作进程\n', desiredWorkers);
    else
        fprintf('并行池已存在，使用 %d 个工作进程\n', pool.NumWorkers);
    end
catch ME
    fprintf('并行池启动失败，使用串行处理: %s\n', ME.message);
end

% 第七步：并行相位解缠处理（核心优化部分）
disp('开始并行相位解缠处理...');
startTime = tic;

% 使用parfor并行处理每个岛屿
parfor i = 1:numObjects
    fprintf('并行处理岛屿 %d/%d\n', i, numObjects);
    
    islandSize = islands(i).Size;
    
    % 为当前岛屿创建临时变量存储结果
    tempUnwrappedPhases = zeros([islandSize, num_interferograms]);
    tempResidualPhases = zeros([islandSize, num_interferograms]);
    
    for interf_idx = 1:num_interferograms
        % 提取当前干涉图的相位数据
        phaseMatrix = islands(i).PhaseMatrixs(:, :, interf_idx);
        
        % 检查数据是否有效（不是全NaN）
        if all(isnan(phaseMatrix(:)))
            tempUnwrappedPhases(:, :, interf_idx) = NaN;
            tempResidualPhases(:, :, interf_idx) = NaN;
            continue;
        end
        
        % 检查相位矩阵是否全零（可能是掩膜问题）
        if all(phaseMatrix(:) == 0)
            tempUnwrappedPhases(:, :, interf_idx) = zeros(islandSize);
            tempResidualPhases(:, :, interf_idx) = zeros(islandSize);
            continue;
        end
        
        % 调用相位解缠函数（警告已被抑制）
        try
            [unwrappedPhase, residual] = unwrap_phase_matrix(phaseMatrix);
            
            % 存储解缠结果
            tempUnwrappedPhases(:, :, interf_idx) = unwrappedPhase;
            tempResidualPhases(:, :, interf_idx) = residual;
            
        catch ME
            fprintf('岛屿 %d 干涉图 %d 解缠失败: %s\n', i, interf_idx, ME.message);
            tempUnwrappedPhases(:, :, interf_idx) = NaN;
            tempResidualPhases(:, :, interf_idx) = NaN;
        end
    end
    
    % 将临时结果存回主结构体
    islands(i).UnWarpedPhases = tempUnwrappedPhases;
    islands(i).ResidualPhases = tempResidualPhases;
end

% 计算并行处理时间
parallelTime = toc(startTime);
fprintf('并行相位解缠完成！耗时: %.2f 秒\n', parallelTime);

% 第八步：多干涉对高程解算（完善接口）
disp('开始多干涉对高程解算...');

% 定义高程解算参数
inclide = 37.9489; % 入射角
R = 633814.6578;   % 卫星到地面的距离，单位m
c = 299792458;     % 光速
lanmuda = c / 5.4050005e+09; % 波长，单位m

% 读取基线信息
bperp_file = '/media/mlxg/d/zjc/tianyi2/INTF/baseline';
bperp = load(bperp_file);

% 参数向量
cansu = [R, inclide, lanmuda];

for i = 1:numObjects
    fprintf('计算岛屿 %d 的高程信息\n', i);
    
    % 提取解缠后的相位数据
    unwrappedPhases = islands(i).UnWarpedPhases;
    
    % 调用高程解算函数
    try
        % 使用多干涉对联合高程解算
        heightMatrix = multiInterferogramHeightEstimation(unwrappedPhases, bperp, cansu, islands(i).TopLeftCorner);
        
        % 存储高程结果
        islands(i).HeightMatrixs = heightMatrix;
        islands(i).Heights = max(heightMatrix(:)) - min(heightMatrix(:)); % 计算高差
        
        fprintf('岛屿 %d 高程解算成功，高差: %.2f m\n', i, islands(i).Heights);
        
    catch ME
        fprintf('岛屿 %d 高程解算失败: %s\n', i, ME.message);
        islands(i).HeightMatrixs = NaN;
        islands(i).Heights = NaN;
    end
end

% 第九步：保存处理结果和统计信息
disp('正在保存处理结果...');

% 显示统计信息
fprintf('\n=== 处理统计信息 ===\n');
fprintf('处理岛屿数量: %d\n', numObjects);
fprintf('干涉图数量: %d\n', num_interferograms);
fprintf('总处理单元: %d\n', numObjects * num_interferograms);

% 计算有效处理数量
valid_count = 0;
for i = 1:numObjects
    for j = 1:num_interferograms
        if ~all(isnan(islands(i).UnWarpedPhases(:, :, j)), 'all')
            valid_count = valid_count + 1;
        end
    end
end
fprintf('成功处理单元: %d\n', valid_count);
fprintf('成功率: %.2f%%\n', 100 * valid_count / (numObjects * num_interferograms));
fprintf('并行处理时间: %.2f 秒\n', parallelTime);

% 保存结果到文件（可选）
results_filename = fullfile(intf_dir, 'multi_interferogram_results_parallel.mat');
save(results_filename, 'islands', 'file_list', 'num_interferograms', 'numObjects', 'parallelTime');

fprintf('结果已保存至: %s\n', results_filename);
fprintf('多干涉对处理完成！\n');

% 关闭并行池
try
    delete(gcp('nocreate'));
    fprintf('并行池已关闭\n');
catch
    fprintf('并行池关闭失败\n');
end

% 修改点2：新增基于值一致性的连通分量标记函数
function [labelMatrix, numComponents] = valueBasedConnectedComponents(matrix, connectivity)
    % 基于值一致性的连通分量标记算法
    % 输入：
    %   matrix: 输入矩阵，包含不同的整数值
    %   connectivity: 连通性（4或8）
    % 输出：
    %   labelMatrix: 标记矩阵，相同值且连通的区域具有相同标签
    %   numComponents: 找到的连通分量数量
    
    [rows, cols] = size(matrix);
    labelMatrix = zeros(rows, cols);
    visited = false(rows, cols);
    numComponents = 0;
    
    % 定义邻域偏移量
    if connectivity == 4
        neighbors = [0, -1; -1, 0; 0, 1; 1, 0]; % 4连通：上下左右
    elseif connectivity == 8
        neighbors = [0, -1; -1, 0; 0, 1; 1, 0; -1, -1; -1, 1; 1, -1; 1, 1]; % 8连通
    else
        error('连通性必须是4或8');
    end
    
    % BFS遍历所有像素
    for i = 1:rows
        for j = 1:cols
            % 跳过已访问或值为0的像素
            if visited(i, j) || matrix(i, j) == 0
                continue;
            end
            
            % 新的连通分量
            numComponents = numComponents + 1;
            currentValue = matrix(i, j);
            
            % BFS队列
            queue = [i, j];
            visited(i, j) = true;
            labelMatrix(i, j) = numComponents;
            
            % BFS遍历
            while ~isempty(queue)
                currentPixel = queue(1, :);
                queue(1, :) = [];
                
                % 检查所有邻居
                for k = 1:size(neighbors, 1)
                    ni = currentPixel(1) + neighbors(k, 1);
                    nj = currentPixel(2) + neighbors(k, 2);
                    
                    % 检查边界
                    if ni >= 1 && ni <= rows && nj >= 1 && nj <= cols
                        % 检查是否未访问且值相同
                        if ~visited(ni, nj) && matrix(ni, nj) == currentValue
                            visited(ni, nj) = true;
                            labelMatrix(ni, nj) = numComponents;
                            queue = [queue; ni, nj];
                        end
                    end
                end
            end
        end
    end
end

% 高程解算函数
function heightMatrix = multiInterferogramHeightEstimation(unwrappedPhases, bperp, cansu, topLeftCorner)
    % 多干涉对高程解算函数
    % 输入：
    %   unwrappedPhases - 解缠相位数据（m×n×N）
    %   bperp - 基线信息
    %   cansu - 参数向量 [R, inclide, lanmuda]
    %   topLeftCorner - 左上角坐标
    
    % 提取参数
    R = cansu(1);
    inclide = cansu(2);
    lanmuda = cansu(3);
    
    % 检查解缠相位数据有效性
    if all(isnan(unwrappedPhases(:)))
        error('所有解缠相位数据均为NaN');
    end
    
    % 将NaN替换为0（无效区域）
    unwrappedPhases(isnan(unwrappedPhases)) = 0;
    
    % 调用LGR_demerror_est函数进行高程估计[1,4](@ref)
    % 注意：这里假设bperp的列结构符合LGR_demerror_est的要求
    [Output] = LGR_demerror_est(cansu, unwrappedPhases, bperp(:, 2:3));
    
    % 返回高程矩阵
    heightMatrix = Output.demerror;
    
    % 应用掩膜，确保无效区域的高程为0
    valid_mask = any(unwrappedPhases ~= 0, 3);
    heightMatrix(~valid_mask) = 0;
end