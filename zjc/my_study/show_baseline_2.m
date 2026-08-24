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
    % Method 2: Use provided hard-coded data directly
    fprintf('File does not exist, using provided hard-coded data\n');
    
    % Use your provided data directly
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

%% 3. Data Filtering - Remove data with absolute baseline value <= 60m
fprintf('\n=== Data Filtering ===\n');
fprintf('Original number of interferometric pairs: %d\n', length(baseline));

% Create filter mask: keep only data where |baseline| > 60
filter_mask = abs(baseline) > 30;

% Apply filter to all data arrays
date_pairs_filtered = date_pairs(filter_mask);
baseline_filtered = baseline(filter_mask);
time_interval_filtered = time_interval(filter_mask);
master_dates_filtered = master_dates(filter_mask);
slave_dates_filtered = slave_dates(filter_mask);
master_datenum_filtered = master_datenum(filter_mask);
slave_datenum_filtered = slave_datenum(filter_mask);
temporal_baseline_filtered = temporal_baseline(filter_mask);

fprintf('Number of pairs after filtering (|baseline| > 30m): %d\n', length(baseline_filtered));
fprintf('Number of pairs removed: %d\n', length(baseline) - length(baseline_filtered));

% Calculate percentage removed
removed_percentage = (length(baseline) - length(baseline_filtered)) / length(baseline) * 100;
fprintf('Percentage removed: %.1f%%\n', removed_percentage);

%% 4. Create Research-Quality Figures
figure('Position', [100, 100, 1200, 800], 'Color', 'white');

% Subplot 1: Perpendicular baseline vs Temporal baseline
subplot(2, 2, 1);
scatter(temporal_baseline_filtered, baseline_filtered, 50, time_interval_filtered, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
colormap(jet);
colorbar;
xlabel('Temporal Baseline (days)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
title('InSAR Interferometric Pair Baseline Distribution (|B| > 30m)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Add zero baseline reference line
hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
hold off;

% Subplot 2: Perpendicular baseline histogram
subplot(2, 2, 2);
histogram(baseline_filtered, 20, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
xlabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title('Perpendicular Baseline Distribution Histogram (|B| > 34m)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Add statistical information
mean_baseline = mean(baseline_filtered);
std_baseline = std(baseline_filtered);
text(0.05, 0.95, sprintf('Mean: %.2f m\nStd: %.2f m', mean_baseline, std_baseline), ...
    'Units', 'normalized', 'FontSize', 10, 'BackgroundColor', 'white');

% Subplot 3: Temporal baseline histogram
subplot(2, 2, 3);
histogram(temporal_baseline_filtered, 15, 'FaceColor', [0.8, 0.4, 0.2], 'EdgeColor', 'k');
xlabel('Temporal Baseline (days)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Frequency', 'FontSize', 12, 'FontWeight', 'bold');
title('Temporal Baseline Distribution Histogram', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Add statistical information
mean_temporal = mean(temporal_baseline_filtered);
std_temporal = std(temporal_baseline_filtered);
text(0.05, 0.95, sprintf('Mean: %.1f days\nStd: %.1f days', mean_temporal, std_temporal), ...
    'Units', 'normalized', 'FontSize', 10, 'BackgroundColor', 'white');

% Subplot 4: Baseline variation over time
subplot(2, 2, 4);
% Sort by master image date
[master_datenum_sorted, sort_idx] = sort(master_datenum_filtered);
baseline_sorted = baseline_filtered(sort_idx);

plot(master_datenum_sorted, baseline_sorted, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Master Image Date', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
title('Perpendicular Baseline Variation Over Time (|B| > 30m)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;

% Set date format
datetick('x', 'yyyy-mm', 'keepticks');

% Add zero baseline reference line
hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
hold off;

%% 5. Add Overall Title and Statistical Information
sgtitle('Tianjin Region Baseline Distribution', 'FontSize', 16, 'FontWeight', 'bold');

% Add statistical information at the bottom of the figure
annotation('textbox', [0.1, 0.01, 0.8, 0.04], ...
    'String', sprintf('Data Statistics: Total pairs = %d (Filtered: |B| > 30m) | Perpendicular baseline range = [%.2f, %.2f] m | Temporal baseline range = [%d, %d] days', ...
    length(baseline_filtered), min(baseline_filtered), max(baseline_filtered), min(temporal_baseline_filtered), max(temporal_baseline_filtered)), ...
    'FontSize', 10, 'HorizontalAlignment', 'center', 'EdgeColor', 'none');

%% 6. Save Figures
% Create save directory (if it doesn't exist)
save_dir = './InSAR_baseline_plots_filtered';
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% Save as high-resolution PNG and PDF
saveas(gcf, fullfile(save_dir, 'InSAR_baseline_analysis_filtered.png'));
saveas(gcf, fullfile(save_dir, 'InSAR_baseline_analysis_filtered.pdf'));

fprintf('Figures saved to: %s\n', save_dir);

%% 7. Display Data Statistics
fprintf('\n=== InSAR Baseline Data Statistics (Filtered: |B| > 30m) ===\n');
fprintf('Total interferometric pairs: %d\n', length(baseline_filtered));
fprintf('Perpendicular baseline statistics:\n');
fprintf('  Minimum: %.2f m\n', min(baseline_filtered));
fprintf('  Maximum: %.2f m\n', max(baseline_filtered));
fprintf('  Mean: %.2f m\n', mean(baseline_filtered));
fprintf('  Standard deviation: %.2f m\n', std(baseline_filtered));
fprintf('Temporal baseline statistics:\n');
fprintf('  Minimum: %d days\n', min(temporal_baseline_filtered));
fprintf('  Maximum: %d days\n', max(temporal_baseline_filtered));
fprintf('  Mean: %.1f days\n', mean(temporal_baseline_filtered));
fprintf('  Standard deviation: %.1f days\n', std(temporal_baseline_filtered));

%% 8. Optional: Create 3D Scatter Plot
figure('Position', [100, 100, 800, 600], 'Color', 'white');
scatter3(temporal_baseline_filtered, baseline_filtered, time_interval_filtered, 30, baseline_filtered, 'filled');
colormap(jet);
colorbar;
xlabel('Temporal Baseline (days)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 12, 'FontWeight', 'bold');
zlabel('Interferometric Interval (days)', 'FontSize', 12, 'FontWeight', 'bold');
title('InSAR Interferometric Pair 3D Baseline Distribution (|B| > 30m)', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
box on;
view(45, 30);

% Save 3D figure
saveas(gcf, fullfile(save_dir, 'InSAR_baseline_3D_filtered.png'));

%% 9. Create Comparison Figure (Original vs Filtered)
figure('Position', [100, 100, 1000, 400], 'Color', 'white');

% Subplot 1: Original data
subplot(1, 2, 1);
scatter(temporal_baseline, baseline, 40, 'b', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
xlabel('Temporal Baseline (days)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('Original Data (All Pairs)', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;
hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
hold off;

% Add threshold lines
hold on;
plot(xlim, [30 30], 'r--', 'LineWidth', 1);
plot(xlim, [-30 -30], 'r--', 'LineWidth', 1);
hold off;

% Subplot 2: Filtered data
subplot(1, 2, 2);
scatter(temporal_baseline_filtered, baseline_filtered, 40, 'r', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 0.5);
xlabel('Temporal Baseline (days)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Perpendicular Baseline (m)', 'FontSize', 11, 'FontWeight', 'bold');
title(sprintf('Filtered Data (|B| > 30m, %d pairs)', length(baseline_filtered)), 'FontSize', 12, 'FontWeight', 'bold');
grid on;
box on;
hold on;
plot(xlim, [0 0], 'k--', 'LineWidth', 1);
hold off;

sgtitle('Data Filtering Comparison: Original vs Filtered (|Baseline| > 30m)', 'FontSize', 14, 'FontWeight', 'bold');

% Save comparison figure
saveas(gcf, fullfile(save_dir, 'InSAR_baseline_comparison.png'));

fprintf('\nAll figures generated successfully!\n');