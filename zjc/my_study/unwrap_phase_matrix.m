function [unwrappedPhase, residual] = unwrap_phase_matrix(phaseMatrix)
    % Constants and parameters
    N_nearest = 9;
    N_knn = 4;
    accu = 1; % unit: mm
    diff = sqrt(2) * accu;
    c = 299792458;
    wavelen = 1000 * c / 5.4050005e+09;
    thre_accu = (4 * pi) * diff / wavelen;
    
    % Initialize phase matrix and derived variables
    [rows, cols] = size(phaseMatrix);
    
    phase_vec = phaseMatrix(:);
    indxx = find(ones(size(phaseMatrix)));
    [azis, ranges] = find(ones(size(phaseMatrix)));
    if rows == 1
    azis = azis';
    ranges = ranges';
    indxx = indxx';
end
    obs_candi = [indxx, ranges, azis, phase_vec];
    % 检查 rows 是否等于 1 或者 cols 是否小于等于 4
    if rows == 1 || cols <= 4
        residual(:) = 0;
        unwrappedPhase = phaseMatrix;
        return;  % 退出函数
    end
    disp('Dividing image into patches!');
    coord_keep = obs_candi(:, 2:3);
    nestpoint = knnsearch(coord_keep, coord_keep, 'k', N_nearest);
    pkeep = size(nestpoint, 1);
    ranges = obs_candi(:, 2);
    azis = obs_candi(:, 3);
    obs_phase = obs_candi(:, end);
    pcluster_range = ranges(nestpoint);
    pcluster_azi = azis(nestpoint);
    pcluster_obs = obs_phase(nestpoint);
    pcluster_indx = nestpoint; 
    arc_perpoint = N_nearest * N_knn;
    
    % Loop through each patch
    disp('Start estimation in each patch');
    pwhole = [];
    resid_keep = [];
    
    % Use parallel for loop to improve performance
    parfor i = 1:pkeep
        % Process each cluster window
        pcluster_indx_each = pcluster_indx(i, :);
        pcluster_obs_win = pcluster_obs(i, :);
        pcluster_range_win = pcluster_range(i, :);
        pcluster_azi_win = pcluster_azi(i, :);

        np = [pcluster_range_win', pcluster_azi_win'];

        % Construct arcs
        IDX = local_knn(np, N_knn, 0);
        IDX_froml = IDX.from;
        IDX_tol = IDX.to;
        narc = length(IDX_froml);
        IDX_from = pcluster_indx_each(IDX_froml);
        IDX_to = pcluster_indx_each(IDX_tol);
        
        % Normalization
        a = 0; b = 1;
        m = min(pcluster_range_win); M = max(pcluster_range_win);
        ranges_norm = (b-a) * (pcluster_range_win - m) / (M - m);
        m = min(pcluster_azi_win); M = max(pcluster_azi_win);
        azis_norm = (b-a) * (pcluster_azi_win - m) / (M - m);

        dr = ranges_norm(IDX_tol) - ranges_norm(IDX_froml);
        dazi = azis_norm(IDX_tol) - azis_norm(IDX_froml);
        drdazi = ranges_norm(IDX_tol).*azis_norm(IDX_tol) - ranges_norm(IDX_froml).*azis_norm(IDX_froml);
        arc_poly = [dr', dazi', drdazi'];

        % Prepare observations
        y = wrap(pcluster_obs_win(IDX_tol) - pcluster_obs_win(IDX_froml));

        % Solve parameters
        [b, stats] = robustfit(arc_poly, y, 'bisquare', 4.685, 'off');
        obse = arc_poly * b;

        % Calculate residuals
        res = stats.resid;

        nn = find(abs(res) < thre_accu);
        IDX_from_keep = IDX_from(nn);
        IDX_to_keep = IDX_to(nn);
        obse_keep = obse(nn, :);
        res_keep = res(nn, :);
        pcluster_keep = [IDX_from_keep', IDX_to_keep', obse_keep];
        pcluster_residual = [IDX_from_keep', IDX_to_keep', res_keep];

        pwhole = [pwhole; pcluster_keep];
        resid_keep = [resid_keep; pcluster_residual];
    end
    
    % Solve for phase unwrapping result
    NARC = length(pwhole(:, 1)); 
    NTCP = pkeep;
    arc_index = (1:NARC)';
    arc_tcp1 = sparse(arc_index, pwhole(:, 1), -1 * ones(NARC, 1), NARC, NTCP);
    arc_tcp2 = sparse(arc_index, pwhole(:, 2), ones(NARC, 1), NARC, NTCP);
    arc_tcp = arc_tcp1 + arc_tcp2;

    clear arc_tcp1 arc_tcp2
    xxx = sum(abs(arc_tcp));
    xxxd = find(xxx ~= 0);
    tempd = arc_tcp(:, xxxd);   
    tempd(:, 1) = [];

    % Compute result
    par_re = lsmr(tempd, pwhole(:, end));
    par_re = [0; par_re];
    %par_re = [par_re;0];
    resi_re = lsmr(tempd, resid_keep(:, end));
    resi_re = [0; resi_re];
    %resi_re = [resi_re;0];
    xxx = nan(size(phaseMatrix));
    resi_xxx = xxx;
    gind = obs_candi(:, 1);
    xxx(gind(xxxd)) = par_re;
    resi_xxx(gind(xxxd)) = resi_re;
    unwrappedPhase = xxx;
    residual = resi_xxx;
end