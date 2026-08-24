function [ arc_interval_ph,unwrap_correct_log_final ] = triplet_phase_closure_mcf_2015( tri2ifg_matrix,ifg_ph_wrap,Input,correct_idx )
%TRIPLET_PHASE_CLOSURE_MCF_2015 (FINAL STABLE)
% Detect and correct unwrapping error at arcs based on triplet closure + MCF (intlinprog)
%
% Stable patch (2026-01):
%   - Robustly enforce ifg_ph_wrap orientation to match tri2ifg_matrix:
%       tri2ifg_matrix is (Ntri x NIFG_expected)
%       ifg_ph_wrap must be (NIFG_expected x NARC)
%     If user passes (NARC x NIFG_expected), auto-transpose.
%   - Ensure the partial matrices used in correction keep the same convention,
%     so that the old line-76 "+" never fails due to dimension mismatch.
%
% Original author: LIANG Hongyu @ PolyU-LSGI (2019)
% Stability patch: ChatGPT (2026-01)

%% check input
narginchk(4,4);
if nargin < 4
    correct_idx = 0;
end

%% ---- Enforce consistent dimensions with tri2ifg_matrix ----
NIFG_expected = size(tri2ifg_matrix,2); % columns of tri2ifg_matrix = #IFG
[ifg_ph_wrap, transposed_main] = enforce_ifg_dims(ifg_ph_wrap, NIFG_expected, 'ifg_ph_wrap'); %#ok<NASGU>

% After enforcement, we guarantee:
%   size(ifg_ph_wrap,1) == NIFG_expected
NIFG = size(ifg_ph_wrap,1);
NARC = size(ifg_ph_wrap,2);

Nintv = Input.NSLC-1;
Ntri  = size(tri2ifg_matrix,1);
B     = Input.tmatrix;

arc_interval_ph = nan(Nintv,NARC);

%% form triplet_closure for all arcs
% (Ntri x NIFG) * (NIFG x NARC) => (Ntri x NARC)
triplet_closure_integer = round( (tri2ifg_matrix * ifg_ph_wrap) / (2*pi) );

%% detect arc with zero closure
triplet_closure_log = (triplet_closure_integer == 0);
unwrap_correct_log  = (sum(triplet_closure_log,1) == Ntri);

%% derive intv phase for arc with zero closure
arc_interval_ph(:,unwrap_correct_log) = B \ ifg_ph_wrap(:,unwrap_correct_log);
unwrap_correct_log_final = unwrap_correct_log;

if correct_idx == 1
    %% detect arc with non-zero closure
    unwrap_partial_idx      = find(~unwrap_correct_log);
    NARC_unwrap_partial     = length(unwrap_partial_idx);

    if NARC_unwrap_partial == 0
        % nothing to correct
        unwrap_correct_log_final = unwrap_correct_log_final';
        arc_interval_ph          = arc_interval_ph.';
        return;
    end

    %% find dataset for arc with non-zero closure
    ifg_ph_wrap_partial = ifg_ph_wrap(:,unwrap_partial_idx);
    % Defensive: ensure partial keeps [NIFG x NARC_partial]
    ifg_ph_wrap_partial = enforce_ifg_dims(ifg_ph_wrap_partial, NIFG_expected, 'ifg_ph_wrap_partial');

    %% corrct the ifg_ph of arc with non-zero closure
    options = optimoptions('intlinprog','Display','off'); % matlab 2015+
    cost  = ones(NIFG*2,1);
    L     = zeros(NIFG*2,1);
    U     = Inf(NIFG*2,1);

    % beq: (Ntri x NIFG) * (NIFG x NARC_partial) => (Ntri x NARC_partial)
    beq = -round( (tri2ifg_matrix * ifg_ph_wrap_partial) / (2*pi) );

    Aeq   = [tri2ifg_matrix, -tri2ifg_matrix];
    intcon = 1:(NIFG*2);

    blockstep = 10000;
    blocknum  = ceil(NARC_unwrap_partial/blockstep);

    beq_cell = cell(blocknum,1);
    K_cell   = cell(1,blocknum);

    for j = 1:blocknum
        pt_sidx = (j-1)*blockstep+1;
        pt_eidx = min(j*blockstep, NARC_unwrap_partial);
        pt_idx  = pt_sidx:pt_eidx;
        beq_cell{j,1} = beq(:,pt_idx);
    end

    for j = 1:blocknum
        beq_temp = beq_cell{j,1};
        NARC_unwrap_partial_temp = size(beq_temp,2);

        xpm = zeros(NIFG*2, NARC_unwrap_partial_temp);

        parfor i = 1:NARC_unwrap_partial_temp
            sol = intlinprog(cost,intcon,[],[],Aeq,beq_temp(:,i),L,U,options);
            if isempty(sol)
                % If infeasible, keep zeros (will likely fail closure later, but won't crash)
                xpm(:,i) = 0;
            else
                xpm(:,i) = sol;
            end
        end

        xpm = round(xpm);
        xp  = xpm(1:NIFG,:);
        xm  = xpm(NIFG+1:end,:);
        K_cell{1,j} = xp - xm; % (NIFG x NARC_block)
    end

    K = cell2mat(K_cell); % should be (NIFG x NARC_unwrap_partial)

    % ---- FINAL STABILITY GUARD for the old line-76 "+" ----
    % Ensure K and ifg_ph_wrap_partial have identical size before adding.
    if ~isequal(size(K), size(ifg_ph_wrap_partial))
        % The only reasonable mismatch here is transpose; try to fix.
        if isequal(size(K), fliplr(size(ifg_ph_wrap_partial)))
            ifg_ph_wrap_partial = ifg_ph_wrap_partial.'; % bring to (NIFG x NARC_partial)
        end
    end

    if ~isequal(size(K), size(ifg_ph_wrap_partial))
        error('Size mismatch before adding 2pi*K: size(K)=%s, size(ifg_ph_wrap_partial)=%s. Check upstream y1_temp orientation.', ...
            mat2str(size(K)), mat2str(size(ifg_ph_wrap_partial)));
    end

    % Replace the fragile "K*(2*pi)+wrap" with a stable equivalent form:
    ifg_ph_partial = ifg_ph_wrap_partial + (2*pi)*K;  % (NIFG x NARC_partial)

    %% form triplet_closure again
    triplet_closure_partial_integer = round( (tri2ifg_matrix * ifg_ph_partial) / (2*pi) );

    %% update intv phase for arc with zero cloosure
    triplet_closure_partial_correct_idx = (triplet_closure_partial_integer == 0);
    unwrap_correct_log_partial_1        = (sum(triplet_closure_partial_correct_idx,1) == Ntri);

    unwrap_correct_idx_partial_2 = unwrap_partial_idx(unwrap_correct_log_partial_1);
    unwrap_correct_log_partial_3 = false(1,NARC);
    unwrap_correct_log_partial_3(unwrap_correct_idx_partial_2) = true;

    arc_interval_ph(:,unwrap_correct_idx_partial_2) = B \ ifg_ph_partial(:,unwrap_correct_log_partial_1);

    %% update IDX and intv phase
    unwrap_correct_log_final = unwrap_correct_log | unwrap_correct_log_partial_3;
end

unwrap_correct_log_final = unwrap_correct_log_final';
arc_interval_ph          = arc_interval_ph.';

if correct_idx == 1
    disp(['Number of arcs:                                    ', num2str(NARC)]);
    disp(['Number of arcs recovered without ambiguity:        ', num2str(sum(unwrap_correct_log))]);
    disp(['Number of arcs recovered with ambiguity:           ', num2str(sum(unwrap_correct_log_partial_1))]);
    disp(['Number of arcs unrecovered with ambiguity:         ', num2str(NARC_unwrap_partial-sum(unwrap_correct_log_partial_1))]);
end
end

%% ======================== subfunction (original, stabilized) =========================
function ifg_ph_partial_correct_temp = ifg_ph_correct_mcf(tri2ifg_matrix,ifg_ph_wrap_partial_temp,options)
NIFG_expected = size(tri2ifg_matrix,2);
ifg_ph_wrap_partial_temp = enforce_ifg_dims(ifg_ph_wrap_partial_temp, NIFG_expected, 'ifg_ph_wrap_partial_temp');

NIFG = size(tri2ifg_matrix,2);
beq  = -round((tri2ifg_matrix * ifg_ph_wrap_partial_temp)/(2*pi));

cost = ones(2*NIFG,1);
L    = zeros(2*NIFG,1);
U    = Inf(2*NIFG,1);
Aeq  = [tri2ifg_matrix, -tri2ifg_matrix];

xpm = linprog(cost,[],[],Aeq,beq,L,U,[],options);
xpm = reshape(round(xpm),[],2);
k   = xpm(:,1)-xpm(:,2);
k   = round(k);

% Stable add
ifg_ph_partial_correct_temp = ifg_ph_wrap_partial_temp + (2*pi)*k;
end

%% ======================== helper: enforce orientation =========================
function [X, didTranspose] = enforce_ifg_dims(X, nIfgExpected, varname)
didTranspose = false;

if isempty(X)
    return;
end

sz = size(X);

% We want size(X,1) == nIfgExpected.
if sz(1) ~= nIfgExpected
    if sz(2) == nIfgExpected
        X = X.';              % transpose to (nIfgExpected x nArc)
        didTranspose = true;
    else
        error('%s size=%s is incompatible with nIfgExpected=%d (from tri2ifg_matrix).', ...
            varname, mat2str(sz), nIfgExpected);
    end
end
end