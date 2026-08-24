
disp('Unwrapping all the interferograms!');
aven = 'H:\TY_tianjing\INTF\ave';
mag=freadbkB(aven,2500,'float32');
%构造一个分别分别存储距离向和方位向的数组，xazis（i）,xranges(i)表示第i个像素的雷达坐标
[xazis,xranges]=find(ones(size(mag)));
mag_db=mag2db(mag);

nlines=1000;
N_nearest=49;
N_knn=9;
accu=1; %unit:mm
diff=sqrt(2)*accu;
c=299792458;wavelen=1000*c/5.4050005e+09;
thre_accu=(4*pi)*diff/wavelen;
unw_phase=[];

inft='H:\TY_tianjing\INTF\20231007-20231029.diff';
image=freadbkB(inft,2500, 'cpxfloat32');
phases=angle(image);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[rows,cols]=size(phases);
phase_vec=phases(:);
indxx=find(ones(size(phases)));
[azis,ranges]=find(ones(size(phases)));
obs_candi=[indxx,ranges,azis,phase_vec];
%clear phases ranges azis indxx phase_vec  删掉强度不显著的像元
nn=mag_db<90;
obs_candi(nn,:)=[];clear nn

disp('Dividing image into patchs!');%%根据邻近像元法分块
coord_keep=obs_candi(:,2:3);
nestpoint=knnsearch(coord_keep,coord_keep,'k',N_nearest);%%size(nestpoint)=(azis-nn,N_nearest)
pkeep=size(nestpoint,1);
ranges=obs_candi(:,2);azis=obs_candi(:,3); obs_phase=obs_candi(:,end);
pcluster_range=ranges(nestpoint);
pcluster_azi=azis(nestpoint);
pcluster_obs=obs_phase(nestpoint);
pcluster_indx=nestpoint; clear nestpoint
arc_perpoint=N_nearest*N_knn;

%%在每个小窗口内进行估计
disp('Start estimation in each patch');
pwhole=[];resid_keep=[];

parfor i=1:pkeep
    
pcluster_indx_each=pcluster_indx(i,:);%%获取每个点的小窗内部雷达坐标与相位值
pcluster_obs_win=pcluster_obs(i,:);
pcluster_range_win=pcluster_range(i,:);
pcluster_azi_win=pcluster_azi(i,:);

np=[pcluster_range_win',pcluster_azi_win'];

 %%构弧段
 IDX=local_knn(np,N_knn,0);%IDX中包括弧段的起始点序数
 IDX_froml=IDX.from;IDX_tol=IDX.to;
 narc=length(IDX_froml);
 IDX_from=pcluster_indx_each(IDX_froml);IDX_to=pcluster_indx_each(IDX_tol);%找到i起始点对于的像素坐标58

 
    a=0;b=1; % scale to [0,1],normalization

    m = min(pcluster_range_win); M = max(pcluster_range_win);
    ranges = (b-a) * (pcluster_range_win-m)/(M-m) ;
    m = min(pcluster_azi_win); M = max(pcluster_azi_win);
    azis= (b-a) * (pcluster_azi_win-m)/(M-m) ;
    
    dr=ranges(IDX_tol)-ranges(IDX_froml);
    dazi=azis(IDX_tol)-azis(IDX_froml);
    drdazi=ranges(IDX_tol).*azis(IDX_tol)-ranges(IDX_froml).*azis(IDX_froml);
    arc_poly=[dr',dazi',drdazi'];
      
      %prepare obs.
    y=wrap(pcluster_obs_win(IDX_tol)-pcluster_obs_win(IDX_froml));

       % solve parameters
   [b,stats]=robustfit(arc_poly,y,'bisquare',4.685,'off');
    obse=arc_poly*b;
       
    res=stats.resid;
    
 
  nn=find(abs(res)< thre_accu);
  IDX_from_keep=IDX_from(nn);
  IDX_to_keep=IDX_to(nn);
  obse_keep=obse(nn,:);
  res_keep=res(nn,:);
  pcluster_keep=[IDX_from_keep',IDX_to_keep',obse_keep];
  pcluster_residual=[IDX_from_keep',IDX_to_keep',res_keep];

  pwhole=[pwhole;pcluster_keep];
  
  resid_keep=[resid_keep;pcluster_residual];
end


NARC=length(pwhole(:,1)); % the number of arcs
NTCP=pkeep;
%  %++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%  % get the design matrix related to par. ( dem_error and linear v) of
%  % all TCPs
%  % the design matrix indicating the relationship between TCPs and arcs

arc_index=(1:NARC)';
arc_tcp1=sparse(arc_index,pwhole(:,1),-1*ones(NARC,1),NARC,NTCP);
arc_tcp2=sparse(arc_index,pwhole(:,2),ones(NARC,1),NARC,NTCP);
arc_tcp=arc_tcp1+arc_tcp2;


clear arc_tcp1 arc_tcp2
xxx=sum(abs(arc_tcp));
xxxd=find(xxx~=0);
tempd=arc_tcp(:,xxxd);   
tempd(:,1)=[];

%par_re=tempd\parc_obse_vector;
par_re=lsmr(tempd, pwhole(:,end));
par_re=[0;par_re];  %%选择第一点作为参考点，解缠得到相对于参考点的相对行为
resi_re=lsmr(tempd,resid_keep(:,end));
resi_re=[0;resi_re];
xxx=nan(size(mag_db));
resi_xxx=xxx;
gind=obs_candi(:,1);
xxx(gind(xxxd))=par_re;
resi_xxx(gind(xxxd))=resi_re;
unw_2022063020220712=xxx;
residual_2022063020220712=resi_xxx;
