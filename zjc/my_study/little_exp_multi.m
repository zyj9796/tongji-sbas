% 多对干涉对高程解算主程序（指定矩形区域版）
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

% 第三步：定义处理区域参数（固定区域）代替原有的mask分析
% ------------------ 修改开始 ------------------
startRow = 3552;    % 起始行
startCol = 6324;    % 起始列
blockRows = 90;     % 区域行数
blockCols = 300;    % 区域列数

fprintf('使用固定矩形区域: Start[%d, %d], Size[%d, %d]\n', startRow, startCol, blockRows, blockCols);

% 强制设置岛屿数量为 1
numObjects = 1;

% 初始化结构体数组
islands = struct('Matrix', {}, 'TopLeftCorner', {}, 'Size', {}, 'Value', {}, ...
                 'PhaseMatrixs', {}, 'UnWarpedPhases', {}, 'ResidualPhases', {}, ...
                 'HeightMatrixs', {}, 'Heights', {});
% ------------------ 修改结束 ------------------

% 第四步：初始化这个单一的“矩形岛屿”
disp('正在初始化岛屿结构...');

% 因为只有一个固定的矩形岛屿，直接赋值，不需要循环
i = 1;

% 孤岛矩阵的大小
sizeOfIsland = [blockRows, blockCols];

% 孤岛矩阵的左上角坐标
topLeftCorner = [startRow, startCol];

% 创建一个全1的掩膜矩阵（代表整个矩形区域都是有效岛屿）
islandMatrix = ones(blockRows, blockCols); 
islandValue = 1; 

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

disp(['岛屿结构初始化完成，模式：单矩形区域']);

% 第五步：读取所有干涉图数据并填充PhaseMatrixs
disp('开始读取干涉图数据...');
for file_idx = 1:num_interferograms
    current_file = fullfile(intf_dir, file_list(file_idx).name);
    disp(['处理干涉图 ', num2str(file_idx), '/', num2str(num_interferograms), ': ', file_list(file_idx).name]);
    
    try
        % 读取复数干涉图数据
        image_data = freadbkB(current_file, lines, 'cpxfloat32');
        phases = angle(image_data); % 获取相位信息
        
        % 针对唯一的岛屿提取数据
        i = 1; 
        topLeft = islands(i).TopLeftCorner;
        islandSize = islands(i).Size;
        % islandMatrix = islands(i).Matrix; % 全是1，这里其实不需要再过滤了
        
        % 计算切片范围
        minRow = topLeft(1);
        minCol = topLeft(2);
        maxRow = minRow + islandSize(1) - 1;
        maxCol = minCol + islandSize(2) - 1;
        
        % 边界检查（防止定义的矩形超出图像范围）
        if maxRow > size(phases, 1) || maxCol > size(phases, 2)
            error('定义的矩形区域超出了图像边界！');
        end

        % 提取相应的phase子矩阵
        phasePatch = phases(minRow:maxRow, minCol:maxCol);
        
        % 存入结构体
        islands(i).PhaseMatrixs(:, :, file_idx) = phasePatch;
        
    catch ME
        fprintf('处理文件失败: %s, 错误: %s\n', current_file, ME.message);
        % 如果某个文件读取失败，将该层的相位数据设为NaN
        islands(1).PhaseMatrixs(:, :, file_idx) = NaN;
    end
end

disp('所有干涉图数据读取完成！');

% 第六步：启动并行池并配置并行环境
% 注意：虽然现在只有1个岛屿，parfor循环只会执行一次，但保留并行结构以便后续扩展或维持代码逻辑
disp('启动并行池用于相位解缠...');
try
    pool = gcp('nocreate');
    if isempty(pool)
        parpool(64); % 根据实际核心数调整
        fprintf('并行池已启动\n');
    else
        fprintf('并行池已存在，使用 %d 个工作进程\n', pool.NumWorkers);
    end
catch ME
    fprintf('并行池启动失败，使用串行处理: %s\n', ME.message);
end

% 第七步：并行相位解缠处理
disp('开始处理相位解缠...');
startTime = tic;

% 这里 numObjects = 1，所以只会循环一次
parfor i = 1:numObjects
    fprintf('正在处理区域 %d/%d\n', i, numObjects);
    
    islandSize = islands(i).Size;
    
    % 为当前岛屿创建临时变量存储结果
    tempUnwrappedPhases = zeros([islandSize, num_interferograms]);
    tempResidualPhases = zeros([islandSize, num_interferograms]);
    
    for interf_idx = 1:num_interferograms
        % 提取当前干涉图的相位数据
        phaseMatrix = islands(i).PhaseMatrixs(:, :, interf_idx);
        
        % 检查数据是否有效
        if all(isnan(phaseMatrix(:)))
            tempUnwrappedPhases(:, :, interf_idx) = NaN;
            tempResidualPhases(:, :, interf_idx) = NaN;
            continue;
        end
        
        if all(phaseMatrix(:) == 0)
            tempUnwrappedPhases(:, :, interf_idx) = zeros(islandSize);
            tempResidualPhases(:, :, interf_idx) = zeros(islandSize);
            continue;
        end
        
        % 调用相位解缠函数
        try
            [unwrappedPhase, residual] = unwrap_phase_matrix(phaseMatrix);
            
            % 存储解缠结果
            tempUnwrappedPhases(:, :, interf_idx) = unwrappedPhase;
            tempResidualPhases(:, :, interf_idx) = residual;
            
        catch ME
            fprintf('区域 %d 干涉图 %d 解缠失败: %s\n', i, interf_idx, ME.message);
            tempUnwrappedPhases(:, :, interf_idx) = NaN;
            tempResidualPhases(:, :, interf_idx) = NaN;
        end
    end
    
    % 将临时结果存回主结构体
    islands(i).UnWarpedPhases = tempUnwrappedPhases;
    islands(i).ResidualPhases = tempResidualPhases;
end

parallelTime = toc(startTime);
fprintf('相位解缠完成！耗时: %.2f 秒\n', parallelTime);

% 第八步：多干涉对高程解算
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
    fprintf('计算区域 %d 的高程信息\n', i);
    
    % 提取解缠后的相位数据
    unwrappedPhases = islands(i).UnWarpedPhases;
    
    % 调用高程解算函数
    try
        heightMatrix = multiInterferogramHeightEstimation(unwrappedPhases, bperp, cansu, islands(i).TopLeftCorner);
        
        % 存储高程结果
        islands(i).HeightMatrixs = heightMatrix;
        islands(i).Heights = max(heightMatrix(:)) - min(heightMatrix(:)); % 计算高差
        
        fprintf('区域 %d 高程解算成功，高差: %.2f m\n', i, islands(i).Heights);
        
    catch ME
        fprintf('区域 %d 高程解算失败: %s\n', i, ME.message);
        islands(i).HeightMatrixs = NaN;
        islands(i).Heights = NaN;
    end
end

% 第九步：保存处理结果和统计信息
disp('正在保存处理结果...');

% 显示统计信息
fprintf('\n=== 处理统计信息 ===\n');
fprintf('处理模式: 固定矩形区域\n');
fprintf('区域大小: %d x %d\n', blockRows, blockCols);
fprintf('干涉图数量: %d\n', num_interferograms);

% 计算有效处理数量
valid_count = 0;
for j = 1:num_interferograms
    if ~all(isnan(islands(1).UnWarpedPhases(:, :, j)), 'all')
        valid_count = valid_count + 1;
    end
end

fprintf('成功处理层数: %d\n', valid_count);

% 保存结果到文件
results_filename = fullfile(intf_dir, 'single_block_results.mat');
% 移除原有的 mask_real 保存需求，因为现在是自定义区域
save(results_filename, 'islands', 'file_list', 'num_interferograms', 'numObjects', 'parallelTime');

fprintf('结果已保存至: %s\n', results_filename);
fprintf('处理完成！\n');

% 关闭并行池
try
    delete(gcp('nocreate'));
    fprintf('并行池已关闭\n');
catch
    fprintf('并行池关闭失败\n');
end

% 高程解算函数 (保持不变)
function heightMatrix = multiInterferogramHeightEstimation(unwrappedPhases, bperp, cansu, topLeftCorner)
    % 多干涉对高程解算函数
    
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
    
    % 调用LGR_demerror_est函数进行高程估计
    [Output] = LGR_demerror_est(cansu, unwrappedPhases, bperp(:, 2:3));
    
    % 返回高程矩阵
    heightMatrix = Output.demerror;
    
    % 应用掩膜，确保无效区域的高程为0
    valid_mask = any(unwrappedPhases ~= 0, 3);
    heightMatrix(~valid_mask) = 0;
end