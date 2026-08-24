%% InSAR Interferometric Pair Baseline Plotting Script - Enhanced Version
% Author: AI Assistant
% Date: 2025-12-14
% Function: Read InSAR baseline data and plot professional baseline charts

clear; clc; close all;

%% 1. Data Reading and Parsing - Enhanced version for irregular spaces
% Assume data file path
data_path = '/media/mlxg/d/zjc/tianyi2/INTF/baseline';

% Method 1: If file exists, read using a more flexible approach
if exist(data_path, 'file')
    fprintf('Reading data from file: %s\n', data_path);
    
    % Read entire file content
    fid = fopen(data_path, 'r');
    file_content = fread(fid, '*char')';
    fclose(fid);
    
    % Split by lines
    lines = strsplit(file_content, '\n');
    
    % Initialize storage arrays
    date_pairs = {};
    baseline = [];
    time_interval = [];
    
    % Parse line by line
    line_count = 0;
    for i = 1:length(lines)
        line = strtrim(lines{i});
        if isempty(line) || all(line == ' ')
            continue; % Skip empty lines
        end
        
        % Use regular expression to match data pattern
        tokens = regexp(line, '(\d{8}-\d{8})\s+([-\d\.]+)\s+(\d+)', 'tokens');
        
        if ~isempty(tokens)
            line_count = line_count + 1;
            date_pairs{line_count, 1} = tokens{1}{1};
            baseline(line_count, 1) = str2double(tokens{1}{2});
            time_interval(line_count, 1) = str2double(tokens{1}{3});
        else
            fprintf('Warning: Line %d format mismatch: %s\n', i, line);
        end
    end
    
    fprintf('Successfully parsed %d lines of data\n', line_count);
    
else
    % Method 2: Directly use provided hard-coded data
    fprintf('File does not exist, using provided hard-coded data\n');
    
    % Directly use your provided data
    date_pairs = {
        '20231007-20231029'; '20231007-20231109'; '20231007-20231120'; '20231007-20231201';
        '20231029-20231109'; '20231029-20231120'; '20231029-20231201'; '20231029-20231212';
        '20231109-20231120'; '20231109-20231201'; '20231109-20231212'; '20231109-20231223';
        '20231120-20231201'; '20231120-20231212'; '20231120-20231223'; '20231120-20240103';
        '20231201-20231212'; '20231201-20231223'; '20231201-20240103'; '20231201-20240114';
        '20231212-20231223'; '20231212-20240103'; '20231212-20240114'; '20231212-20240125';
        '20231223-20240103'; '20231223-20240114'; '20231223-20240125'; '20231223-20240205';
        '20240103-20240114'; '20240103-20240125'; '20240103-20240205'; '20240103-20240216';
        '20240114-20240125'; '20240114-20240205'; '20240114-20240216'; '20240114-20240227';
        '20240125-20240205'; '20240125-20240216'; '20240125-20240227'; '20240125-20240309';
        '20240205-20240216'; '20240205-20240227'; '20240205-20240309'; '20240205-20240331';
        '20240216-20240227'; '20240216-20240309'; '20240216-20240331'; '20240216-20240411';
        '20240227-20240309'; '20240227-20240331'; '20240227-20240411'; '20240227-20240422';
        '20240309-20240331'; '20240309-20240411'; '20240309-20240422'; '20240309-20240503';
        '20240331-20240411'; '20240331-20240422'; '20240331-20240503'; '20240331-20240514';
        '20240411-20240422'; '20240411-20240503'; '20240411-20240514'; '20240411-20240525';
        '20240422-20240503'; '20240422-20240514'; '20240422-20240525'; '20240422-20240605';
        '20240503-20240514'; '20240503-20240525'; '20240503-20240605'; '20240514-20240525';
        '20240514-20240605'; '20240525-20240605'
    };
    
    baseline = [
        -31.0311; 31.8504; -7.1064; -17.3900; 62.8824; 23.9260; 13.6424; 59.4929;
        -38.8458; -49.1651; -3.3708; -109.5231; -10.2834; 35.5662; -70.6315; 48.0178;
        45.8498; -60.3496; 58.3021; 0.9254; -106.1517; 12.4690; -44.8712; -69.5713;
        118.6573; 61.2838; 36.5886; 14.5523; -57.3754; -82.0701; -104.0995; -89.7032;
        -24.6954; -46.7279; -32.3336; 6.1228; -22.0345; -7.6377; 30.8175; -47.7201;
        14.3964; 52.8538; -25.6850; 38.3713; 38.4561; -40.0830; 23.9732; -42.7729;
        -78.5356; -14.4826; -81.2237; -84.5247; 64.0584; -2.6925; -5.9943; 3.9295;
        -66.7434; -70.0445; -60.1277; 115.6578; -3.3002; 6.6214; 182.4055; 3.7948;
        9.9253; 185.7160; 7.0946; -60.2357; 175.7891; -2.8294; -70.1563; -178.6029;
        -245.9152; -67.3283
    ];
    
    time_interval = [
        22; 33; 44; 55; 11; 22; 33; 44; 11; 22; 33; 44; 11; 22; 33; 44;
        11; 22; 33; 44; 11; 22; 33; 44; 11; 22; 33; 44; 11; 22; 33; 44;
        11; 22; 33; 44; 11; 22; 33; 44; 11; 22; 33; 55; 11; 22; 44; 55;
        11; 33; 44; 55; 22; 33; 44; 55; 11; 22; 33; 44; 11; 22; 33; 44;
        11; 22; 33; 44; 11; 22; 33; 11; 22; 11
    ];
end

%% 2. Data Preprocessing
% Extract master image dates (first 8 digits of each date pair)
master_dates = cell(length(date_pairs), 1);
slave_dates = cell(length(date_pairs), 1);

for i = 1:length(date_pairs)
    dates = strsplit(date_pairs{i}, '-');
    master_dates{i} = dates{1};
    slave_dates{i} = dates{2};
end

% Convert date strings to MATLAB date format
master_datenum = datenum(master_dates, 'yyyymmdd');
slave_datenum = datenum(slave_dates, 'yyyymmdd');

% Calculate temporal baseline (days)
temporal_baseline = slave_datenum - master_datenum;

%% 3. Create Professional Scientific Graphics
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Subplot 1: Perpendicular Baseline vs Temporal Baseline
subplot(2, 2, 1);
scatter(temporal_baseline, baseline, 50, time_interval, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
colormap(jet);
colorbar;
xlabel('Temporal Baseline (days)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
title('InSAR Interferometric Pair Baseline Distribution', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Add zero baseline reference line
hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
hold off;

% Subplot 2: Perpendicular Baseline Histogram
subplot(2, 2, 2);
histogram(baseline, 20, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
xlabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title('Perpendicular Baseline Distribution Histogram', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Add statistical information
mean_baseline = mean(baseline);
std_baseline = std(baseline);
text(0.05, 0.95, sprintf('Mean: %.2f m\nStd Dev: %.2f m', mean_baseline, std_baseline), ...
    'Units', 'normalized', 'FontSize', 10, 'BackgroundColor', 'white');

% Subplot 3: Temporal Baseline Histogram
subplot(2, 2, 3);
histogram(temporal_baseline, 15, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'k');
xlabel('Temporal Baseline (days)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title('Temporal Baseline Distribution Histogram', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Add statistical information
mean_temporal = mean(temporal_baseline);
std_temporal = std(temporal_baseline);
text(0.05, 0.95, sprintf('Mean: %.1f days\nStd Dev: %.1f days', mean_temporal, std_temporal), ...
    'Units', 'normalized', 'FontSize', 10, 'BackgroundColor', 'white');

% Subplot 4: Baseline Variation Over Time
subplot(2, 2, 4);
% Sort by master image date
[master_datenum_sorted, sort_idx] = sort(master_datenum);
baseline_sorted = baseline(sort_idx);

plot(master_datenum_sorted, baseline_sorted, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Master Image Date', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
title('Perpendicular Baseline Variation Over Time', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Set date format
datetick('x', 'yyyy-mm', 'keepticks');

% Add zero baseline reference line
hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
hold off;

%% 4. Add Overall Title and Statistical Information
sgtitle('Tianjin Region Extreme Distribution', 'FontSize', 16, 'FontWeight', 'bold');

% Add statistical information at the bottom of the figure
annotation('textbox', [0.1, 0.01, 0.8, 0.04], ...
    'String', sprintf('Data Statistics: Total Interferometric Pairs = %d | Perpendicular Baseline Range = [%.2f, %.2f] m | Temporal Baseline Range = [%d, %d] days', ...
    length(baseline), min(baseline), max(baseline), min(temporal_baseline), max(temporal_baseline)), ...
    'FontSize', 10, 'HorizontalAlignment', 'center', 'EdgeColor', 'none');

%% 5. Save Graphics
% Create save directory (if it doesn't exist)
save_dir = './InSAR_baseline_plots';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Save as high-resolution PNG and PDF
saveas(gcf, fullfile(save_dir, 'InSAR_baseline_analysis.png'));
saveas(gcf, fullfile(save_dir, 'InSAR_baseline_analysis.pdf'));

fprintf('Graphics saved to: %s\n', save_dir);

%% 6. Display Data Statistics
fprintf('\n=== InSAR Baseline Data Statistics ===\n');
fprintf('Total interferometric pairs: %d\n', length(baseline));
fprintf('Perpendicular baseline statistics:\n');
fprintf('  Minimum: %.2f m\n', min(baseline));
fprintf('  Maximum: %.2f m\n', max(baseline));
fprintf('  Mean: %.2f m\n', mean(baseline));
fprintf('  Standard deviation: %.2f m\n', std(baseline));
fprintf('Temporal baseline statistics:\n');
fprintf('  Minimum: %d days\n', min(temporal_baseline));
fprintf('  Maximum: %d days\n', max(temporal_baseline));
fprintf('  Mean: %.1f days\n', mean(temporal_baseline));
fprintf('  Standard deviation: %.1f days\n', std(temporal_baseline));

%% 7. Optional: Create 3D Scatter Plot
figure('Position', [100, 100, 800, 600], 'Color', 'white');
scatter3(temporal_baseline, baseline, time_interval, 60, baseline, 'filled');
colormap(jet);
colorbar;
xlabel('Temporal Baseline (days)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Interferometric Interval (days)', 'FontSize', 12, 'FontWeight', 'bold');
title('InSAR Interferometric Pair 3D Baseline Distribution', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
view(45, 30);

% Save 3D plot
saveas(gcf, fullfile(save_dir, 'InSAR_baseline_3D.png'));

fprintf('\nAll graphics generation completed!\n');