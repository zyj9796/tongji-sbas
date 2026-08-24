%data = islands(32).PhaseMatrixs(:,:,12);
%data = islands(32).UnWarpedPhases(:,:,12);
data = island.UnWarpedPhase;
% 定义要保留的矩形区域边界
x1 = 50; y1 = 15;
x2 = 260; y2 = 65;
flag = false;
% 创建掩模，将矩形区域外的部分设为0
if flag
mask = zeros(size(data));
mask(y1:y2, x1:x2) = 1;
data = data .* mask;
end
% 将NaN值替换为0[1,2,3](@ref)
data(isnan(data)) = 0;

% 显示结果
figure;
set(gcf, 'Position', [100, 100, 800, 600]);
imagesc(data);
