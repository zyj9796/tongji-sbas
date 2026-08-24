clc
clear all


%% 数据输入，修改

% 入射角
inclide= 43.8;

% 卫星到地面的距离,单位m
R=931355.7668;

% 波长,单位m
lanmuda=0.056;
% 每个干涉对基线信息
bperp=load('intfnotion.txt');

%干涉对相位信息读入
data=ImgReadgeotiff('./UNW_TIF_XZ/');
imagesc(data.datastack(:,:,35))
%% 高程信息估计
cansu=[R,inclide,lanmuda];

[Output]=LGR_demerror_est(cansu,data.datastack,bperp);
figure(1)
imagesc(Output.demerror),colorbar,caxis([-50 50]);
title('高程（m）')
%结果输出
geotiffwrite('dem.tif',Output.demerror,data.R)