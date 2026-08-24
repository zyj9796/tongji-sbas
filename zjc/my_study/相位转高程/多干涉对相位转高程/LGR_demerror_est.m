function  [Output]=LGR_demerror_est(cansu,UNW_diff_pair,bperp,thod);
%%普通DEM误差估计
% cansu:参数信息
% UNW_diff_pair：解缠相位
%bperp：垂直基线和时间基线
%thod：设置估计地表高程最小干涉对个数
if nargin < 4
    thod=size(UNW_diff_pair,3)-1;
end
if thod>size(UNW_diff_pair,3)-1
    thod=size(UNW_diff_pair,3)-1;
end
R_center=cansu(1);   %%%%卫星到地面观测点的距离R
incidence_angle=cansu(2);  %%%%卫星入射角
lanmuda=cansu(3);  %%%%%波长
FEN_MU=R_center*sind(incidence_angle);
[row,col,hgt]=size(UNW_diff_pair);

% UNW_UNW=UNW_diff_pair;

% intf_no=cal_Interfinformation(bperp(:,[2 3]));
%构建设计矩阵
inf_number=size(bperp,1);
BB_all=zeros(inf_number,2);
for ii=1:inf_number
   BB_all(ii,1)=((4*pi*bperp(ii,1))/(lanmuda*FEN_MU));
   BB_all(ii,2)=((4*pi*bperp(ii,2))/(lanmuda));  
end
DEM_error_result=zeros(row,col);
Deformatiaon_rate=zeros(row,col);
% NNA=pinv( BB_all);
for mm=1:row
    mm
    for nn=1:col
   
        L=squeeze(UNW_diff_pair(mm,nn,:));
        index=find(L~=0);
        if size(index,1)>thod
        L_c=L( index);
        BB=BB_all(index,:);
        
        W=pinv(BB)*L_c;

        DEM_error_result(mm,nn)=W(1);
        Deformatiaon_rate(mm,nn)=W(2)*365;
        end
    end
end
Output.demerror=-DEM_error_result;
Output.rate=Deformatiaon_rate;