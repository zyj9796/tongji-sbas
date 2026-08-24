filename = '20231029-20240422.diff';
lines = 7500;

r0 = 1;  % 起始行（从1开始）
rN = 2000;  % 结束行
c0 = 1;  % 起始列（从1开始）
cN = 2000;  % 结束列

% 尝试不同的数据格式
formats = {'float32', 'int16', 'uint16', 'double', 'cpxfloat32', 'cpxint16'};

for i = 1:length(formats)B(filename, lines, bkformat, r0, rN, c0, cN);
        fprintf('Successfully read %d elements with format: %s\n', count, bkformat);
        disp(data(1:min(end, 10)));  % 显示前10个数据，确认成功读入
    catch ME
        fprintf('Failed to read with format: %s\n', bkformat);
        disp(ME.message);
    end
end