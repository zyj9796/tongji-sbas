clear
clc
dem_name='Lijiang_DEM.dem';
dem_par_name='Lijiang_DEM.par';
[DEM11,R11]=geotiffread('ALPSMLC30_N027E099_DSM.tif');
[DEM12,R12]=geotiffread('ALPSMLC30_N027E100_DSM.tif');
[DEM13,R13]=geotiffread('ALPSMLC30_N027E101_DSM.tif');
[DEM21,R21]=geotiffread('ALPSMLC30_N026E099_DSM.tif');
[DEM22,R22]=geotiffread('ALPSMLC30_N026E100_DSM.tif');
[DEM23,R23]=geotiffread('ALPSMLC30_N026E101_DSM.tif');

Lat=[R11.Latlim,R12.Latlim,R13.Latlim,R21.Latlim,R22.Latlim,R23.Latlim];
Lon=[R11.Lonlim,R12.Lonlim,R13.Lonlim,R21.Lonlim,R22.Lonlim,R23.Lonlim];

upleftCorner_lat=max(Lat);
upleftCorner_lon=min(Lon);
size_dem=R11.RasterSize;width=size_dem(1);lines=size_dem(2);
merge_width=3*width;
merge_lines=2*lines;
DEM=[DEM11,DEM12,DEM13;
    DEM21,DEM22,DEM23];

DEM=double(DEM);
DEM(DEM==-32768)=0;
DEM(DEM<=0)=0.1;


% fwritebkB(DEM,dem_name,'int16');
fwritebkB(DEM,dem_name,'float32');
fid=fopen(dem_par_name,'w');
fprintf(fid, 'Gamma DIFF&GEO DEM/MAP parameter file\n');
fprintf(fid, '%s', 'title: ');
fprintf(fid, '\nDEM_projection:     EQA\n');
% fprintf(fid, 'data_format:        INTEGER*2\n');
fprintf(fid, 'data_format:        REAL*4\n');
fprintf(fid, 'DEM_hgt_offset:          0.00000\n');
fprintf(fid, 'DEM_scale:               1.00000\n');
fprintf(fid, 'width:                ');
fprintf(fid, '%5d\n', merge_width);
fprintf(fid, 'nlines:               ');
fprintf(fid, '%5d\n', merge_lines);
fprintf(fid, 'corner_lat:');
fprintf(fid, '%16.7f',  upleftCorner_lat);
fprintf(fid, '  decimal degrees\n');
fprintf(fid, 'corner_lon:');
fprintf(fid, '%16.7f', upleftCorner_lon);
fprintf(fid, '  decimal degrees\n');
post_lat=strcat( num2str(R11.DeltaLat),'   ',' decimal degrees\n');
post_lon=strcat(num2str(R11.DeltaLon),'   ',' decimal degrees\n\n');
fprintf(fid, 'post_lat: ');fprintf(fid, '%c',' ');fprintf(fid, post_lat);
fprintf(fid, 'post_lon: ');fprintf(fid, '%c',' ');fprintf(fid, post_lon);

fprintf(fid, 'ellipsoid_name: WGS 84\n');
fprintf(fid, 'ellipsoid_ra:        6378137.000   m\n');
fprintf(fid, 'ellipsoid_reciprocal_flattening:  298.2572236\n\n');

fprintf(fid, 'datum_name: WGS 1984\n');
fprintf(fid, 'datum_shift_dx:              0.000   m\n');
fprintf(fid, 'datum_shift_dy:              0.000   m\n');
fprintf(fid, 'datum_shift_dz:              0.000   m\n');
fprintf(fid, 'datum_scale_m:         0.00000e+00\n');
fprintf(fid, 'datum_rotation_alpha:  0.00000e+00   arc-sec\n');
fprintf(fid, 'datum_rotation_beta:   0.00000e+00   arc-sec\n');
fprintf(fid, 'datum_rotation_gamma:  0.00000e+00   arc-sec\n');
fprintf(fid, 'datum_country_list Global Definition, WGS84, World\n');

fclose(fid);
%EOF