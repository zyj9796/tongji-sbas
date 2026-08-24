function [unwrapped_phase, quality_map] = GoldsteinUnwrap2D(IM, mask)
% GOLDSTEINUNWRAP2D 二维Goldstein枝切法相位解包裹算法
%
% 语法：
%   [unwrapped_phase, quality_map] = GoldsteinUnwrap2D(IM)
%   [unwrapped_phase, quality_map] = GoldsteinUnwrap2D(IM, mask)
%
% 输入：
%   IM - 复数矩阵，包含包裹相位信息
%   mask - 二值掩膜矩阵，指定需要解包裹的区域（可选，默认全1矩阵）
%
% 输出：
%   unwrapped_phase - 解包裹后的相位图像
%   quality_map - 相位质量图（此处使用幅度图像作为质量图）
%
% 内部参数设置：
%   max_box_radius - 最大搜索框半径（像素，默认4）
%   threshold_std - 用于幅度图像阈值化的噪声标准差数（默认5）
%   display_results - 是否显示结果图形（默认false，避免自动弹出图形）
%
% 参考文献：
%   1. Goldstein, R. M., Zebken, H. A., & Werner, C. L. (1988). 
%      Satellite radar interferometry: Two-dimensional phase unwrapping. 
%      Radio Science, 23(4), 713-720.
%   2. Ghiglia, D. C., & Pritt, M. D. (1998). 
%      Two-Dimensional Phase Unwrapping: Theory, Algorithms and Software. 
%      Wiley-Interscience.

% 设置内部参数（不再通过输入参数配置）
max_box_radius = 4;    % 最大搜索框半径
threshold_std = 5;     % 噪声标准差阈值
display_results = false; % 是否显示结果（默认不显示，避免干扰）

% 处理输入参数
if nargin < 2 || isempty(mask)
    % 如果没有提供mask，创建全1掩膜
    mask = ones(size(IM));
end

% 检查输入矩阵和掩膜尺寸是否一致
if ~isequal(size(IM), size(mask))
    error('输入矩阵和掩膜矩阵尺寸不一致');
end

% 计算幅度和相位图像
IM_mag = abs(IM);        % 幅度图像
IM_phase = angle(IM);    % 包裹相位图像

% 执行相位解包裹
residue_charge = PhaseResidues(IM_phase, mask);                       % 计算相位残差
branch_cuts = BranchCuts(residue_charge, max_box_radius, mask);        % 放置枝切线
[IM_unwrapped, rowref, colref] = FloodFill(IM_phase, branch_cuts, mask); % 洪水填充相位解包裹

% 准备输出
unwrapped_phase = IM_unwrapped;
quality_map = IM_mag; % 使用幅度图像作为质量图

% 可选的结果显示
if display_results
    figure; 
    imagesc(residue_charge), colormap(jet), axis square, axis off, title('Phase residues (charged)');
    
    figure; 
    imagesc(branch_cuts), colormap(jet), axis square, axis off, title('Branch cuts');
    
    figure; 
    imagesc(immultiply(IM_phase, mask)), colormap(jet), axis square, axis off, title('Wrapped phase');
    
    % 处理掩膜区域以便更好地显示
    tempmin = min(min(IM_unwrapped(mask ~= 0)));          
    temp = (IM_unwrapped == 0);
    temp_IM = IM_unwrapped;
    temp_IM(temp) = tempmin;
    
    figure; 
    imagesc(temp_IM), colormap(jet), axis square, axis off, title('Unwrapped phase');
    colorbar;
end
end