
% path = '/home/lizzie/OneDrive/data/collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901/c45_01_20'; % a .mat file (but don't write .mat here)
path = 'C:\Users\ea-stone\Documents\data\collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901\c30_01_15'; % a .mat file (but don't write .mat here)

file_to_convert = strcat(path, '.mat');
load(file_to_convert)

file_to_save_to = strcat(path, '.json');

the_file = fopen(file_to_save_to ,'w');
fprintf(the_file, jsonencode(data));
fclose(the_file);
disp("done")