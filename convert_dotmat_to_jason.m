

load('/home/lizzie/OneDrive/data/collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901/c45_01_20.mat')


% create file of metadata 
info_file = fopen('/home/lizzie/OneDrive/data/collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901/all_data.json','w');
fprintf(info_file, jsonencode(data));
fclose(info_file);