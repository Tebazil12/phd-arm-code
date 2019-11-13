higher_path = '/home/lizzie/OneDrive/data/';
file_name = '/all_data.mat';

%% Using Dissimilarity Measure
% current_folder = 'robotcode_first_adapt2019-02-15_1547'; shape = "banana"; n_taps_per_line = 21; %banana

% current_folder = 'robotcode_first_adapt2019-02-15_1547'; shape = "banana"; n_taps_per_line = 21; %banana 21pts
% current_folder = 'robotcode_first_adapt2019-02-15_1515'; shape = "brick"; n_taps_per_line = 21; %brick 21pts
% current_folder = 'robotcode_first_adapt2019-02-15_1156'; shape = "square"; n_taps_per_line = 21; %square 21pts


% current_folder = 'robotcode_first_adapt2019-02-15_1442'; shape = "flower"; n_taps_per_line = 6; %flower pt6
% current_folder = 'robotcode_first_adapt2019-02-15_1436'; shape = "circle"; n_taps_per_line = 6; %circle pt6
% current_folder = 'robotcode_first_adapt2019-02-15_1426'; shape = "flower"; n_taps_per_line = 11; %flower pt11
% current_folder = 'robotcode_first_adapt2019-02-15_1419'; shape = "circle"; n_taps_per_line = 11; %circle pt11
% current_folder = 'robotcode_first_adapt2019-02-15_1406'; shape = "flower"; n_taps_per_line = 21; %flower pt21
% current_folder = 'robotcode_first_adapt2019-02-15_1357'; shape = "circle"; n_taps_per_line = 21; %circle pt21

% current_folder = 'robotcode_first_adapt2019-02-15_1406'; shape = "flower"; n_taps_per_line = 21; %flower step5
% current_folder = 'robotcode_first_adapt2019-02-14_1116'; shape = "flower"; n_taps_per_line = 21; %flower step10
% current_folder = 'robotcode_first_adapt2019-02-15_1357'; shape = "circle"; n_taps_per_line = 21; %circle step5
% current_folder = 'robotcode_first_adapt2019-02-14_1110'; shape = "circle"; n_taps_per_line = 21; %circle step10
% current_folder = 'robotcode_first_adapt2019-02-14_1427'; shape = "circle"; n_taps_per_line = 21; %circle step15
% current_folder = 'robotcode_first_adapt2019-02-14_1443'; shape = "circle"; n_taps_per_line = 21; %circle step20

%% Using GPLVM Only 


% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_1950'; shape = "circle";n_taps_per_line = 21;%NODISS circle
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2002'; shape = "circle";n_taps_per_line = 21;%NODISS circle
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2029'; shape = "banana";n_taps_per_line = 21;%NODISS banana
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2039'; shape = "flower";n_taps_per_line = 21;%NODISS flower
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2051'; shape = "flower";n_taps_per_line = 6;%NODISS flower 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1721'; shape = "flower";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1728'; shape = "flower";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1737'; shape = "flower";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1750'; shape = "flower";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1744'; shape = "flower";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1759'; shape = "circle";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1817'; shape = "circle";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1805'; shape = "circle";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1824'; shape = "circle";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1851'; shape = "banana";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1931'; shape = "banana";n_taps_per_line = 11;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1954'; shape = "brick";n_taps_per_line = 21;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2021'; shape = "brick";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2006'; shape = "brick";n_taps_per_line = 11;%NODISS circle 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1547'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1641'; shape = "banana";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1715'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1659'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1650'; shape = "banana";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1633'; shape = "banana";n_taps_per_line = 6;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1613'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1603'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1555'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1530'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1458'; shape = "brick";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1449'; shape = "brick";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1425'; shape = "brick";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1417'; shape = "brick";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1411'; shape = "brick";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1402'; shape = "brick";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1411'; shape = "brick";n_taps_per_line = 5;%NODISS circle 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1344'; shape = "circle";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1355'; shape = "circle";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1420'; shape = "circle";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1442'; shape = "flower";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1519'; shape = "flower";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1746'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1642'; shape = "banana";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1609'; shape = "banana";n_taps_per_line = 3;%NODISS circle 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM_3refs2019-11-12_1505'; shape = "circle";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM_3refs2019-11-12_1450'; shape = "circle";n_taps_per_line = 5;%NODISS circle 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM_3refs2019-11-13_1224'; shape = "circle";n_taps_per_line = 5;%NODISS circle 6pt
current_folder = 'runrobot_2d_contouring_NO_DISSIM_3refs2019-11-13_1324'; shape = "flower";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM_3refs2019-11-13_1332'; shape = "flower";n_taps_per_line = 5;%NODISS circle 6pt



%% failure tests
% file_name = '/matlab.mat';
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1537'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2013'; shape = "brick";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1837'; shape = "banana";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1830'; shape = "banana";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1857'; shape = "banana";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1845'; shape = "banana";n_taps_per_line = 6;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1908'; shape = "banana";n_taps_per_line = 11;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1905'; shape = "banana";n_taps_per_line = 11;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1915'; shape = "banana";n_taps_per_line = 11;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1923'; shape = "banana";n_taps_per_line = 11;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1927'; shape = "banana";n_taps_per_line = 11;%NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1940'; shape = "banana";n_taps_per_line = 21;%NODISS flower 6pt ...
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2006'; shape = "brick";n_taps_per_line = 5;%NODISS flower 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1522'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1542'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-04_1707'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1720'; shape = "banana";n_taps_per_line = 5;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1558'; shape = "banana";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1556'; shape = "banana";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1547'; shape = "banana";n_taps_per_line = 3;%NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-05_1514'; shape = "flower";n_taps_per_line = 3;%NODISS circle 6pt

full_path = strcat(higher_path, current_folder, file_name);
load(full_path)

figure(1)
clf

figure(1)
subplot(1,4,1)
plot_actual_locations

figure(1)
subplot(1,4,2)
show_gplvm_model

subplot(1,4,3)
show_gplvm_model
view([0,1,0])
% daspect([1 0.25 1])
% daspect([1 0.1 2.2])

subplot(1,4,4)
show_gplvm_model
view([0,0,1])
% daspect([1 0.25 1])
set(gca, 'xminorgrid','on')


print('-dpng', ['/home/lizzie/OneDrive/matlab-figs/', current_folder,'ALL.png']);
savefig(['/home/lizzie/OneDrive/matlab-figs/', current_folder,'ALL.fig']);

show_all_taps_mu_disp_predicts