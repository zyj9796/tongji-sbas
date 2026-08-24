function elevation = phaseToElevation(unwrappedPhase, topleftCorner)
    % 相位转高程plus版（插值计算卫星斜距）
    lines=600;
    col=1000;
    % 雷达参数
    c = 299792458;
    lambda = c / 5.4050005e+09;
    incidenceAngle = deg2rad(37.9489); % 入射角转弧度
    
    % 斜距参数 (需替换为实际值)
    near_range_slc = 633814.6578;   % 近距
    far_range_slc = 638809.3928;    % 远距
    B_perp = 175.7408;            % 基线方向可能需要负号
    
    % 计算斜距增量
    delta_R = (far_range_slc - near_range_slc) / (col - 1);
    
    % 获取当前相位矩阵的左上角在全局图像中的列位置
    start_col = topleftCorner(1);  % 假设topleftCorner格式为 [行,列]
    
    % 预计算转换因子矩阵
    [rows, cols_local] = size(unwrappedPhase);
    elevation = zeros(size(unwrappedPhase));
    
    % 逐像素计算高程
    for r = 1:rows
        for c_local = 1:cols_local
            % 计算全局列索引
            global_col = start_col + c_local - 1;
            
            % 计算当前像素斜距
            R = near_range_slc + (global_col - 1) * delta_R;
            
            % 计算相位到高程的转换因子
            conversion_factor = lambda * R * sin(incidenceAngle) / (4 * pi * B_perp);
            
            % 计算高程 (注意相位符号与基线方向的关系)
            elevation(r, c_local) = conversion_factor * unwrappedPhase(r, c_local);
        end
    end
    
    % 可选：添加参考点高程偏移 (若参考点高程不为0)
    % ref_elevation = 50; 
    % elevation = elevation + ref_elevation;
    elevation = fliplr(elevation);
end