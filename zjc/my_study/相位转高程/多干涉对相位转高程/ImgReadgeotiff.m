function Data=ImgReadgeotiff(imgpath,suffixname)
% % %Batch read geotif of wgs84 coordinates
%%% imgpath:Folders containing geotiff(包含geotiff的文件夹)
%%%suffixname:Suffixes for tif files, default is 'tif'(tif文件的后缀,默认为‘tif’),
if nargin < 2
    suffixname='tif';
end

if nargin < 1
    help ImgReadgeotiff
    return;
end

tag_files = dir([imgpath,'*',suffixname]);
img_num = length(tag_files);
disp(['The number of the ', suffixname,' images:' num2str(img_num)]);

for ii=1:img_num
    tic;
    [Data.datastack(:,:,ii),R]=geotiffread([imgpath,tag_files(ii).name]);
    temp=regexp(tag_files(ii).name,'\d+','match');

    if length(temp)==4     %%%read from SARscape geo tif
        Data.filename(ii,1)=str2double(temp{1});
        Data.filename(ii,2)=str2double(temp{3});
    elseif length(temp)==2 %%%read from GAMMA geo tif
        Data.filename(ii,1)=str2double(temp{1});
        Data.filename(ii,2)=str2double(temp{2});
    elseif length(temp)==1 %%%read from time series geo tif
        Data.filename(ii,1)=str2double(temp{1});
    else
        error('The format of file name should be: <yyyymmdd> or <yyyymmdd_yyyymmdd>or<IS_<yyyymmdd>_m_<number>_<yyyymmdd>_s_<number>_upha_geo>.')
    end
time=toc;
fprintf('Reading Img %d / %d, time = %.0f sec\n',ii,img_num,time);
end
 Data.R=R;
end



