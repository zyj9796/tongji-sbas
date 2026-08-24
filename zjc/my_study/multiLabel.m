function [labelMatrix, numLabels] = multiLabel(matrix, connectivity)
    % 获取所有唯一非零值
    uniqueValues = unique(matrix(matrix ~= 0)); 
    numLabels = 0;
    labelMatrix = zeros(size(matrix));
    
    % 对每个唯一值单独处理
    for v = uniqueValues'
        % 生成当前值的二值掩膜
        binaryMask = (matrix == v);
        
        % 使用bwlabel处理该层
        [currLabels, n] = bwlabel(binaryMask, connectivity);
        
        % 合并标签（偏移量避免重复）
        currLabels(currLabels > 0) = currLabels(currLabels > 0) + numLabels;
        labelMatrix = labelMatrix + currLabels;
        numLabels = numLabels + n;
    end
end