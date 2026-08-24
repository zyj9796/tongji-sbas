clc
clear all
%% 数据输入，修改

% 入射角
inclide= 43.8;

% 卫星到地面的距离,单位m
R=931355.7668;

% 波长,单位m
lanmuda=0.056;

%垂直基线,单位m
bperp=50;

%输入数据相位
%这里用模拟解缠相位值
unw=abs(peaks(100));
figure(1)
imagesc(unw),colorbar,caxis([-8 8]);
title('模拟相位')
%% 相位转高程

%转换系数
zhxs=(4*pi*bperp)/(lanmuda*sind(inclide)*R);
%高程
hgt=unw/zhxs;
figure(1)
imagesc(hgt),colorbar,caxis([-180 180]);
title('高程（m）')








