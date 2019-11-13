%% Using Dissimilarity Measure
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1547/all_data.mat'); n_taps_per_line = 21; %banana

% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1547/all_data.mat'); n_taps_per_line = 21; %banana 21pts
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1515/all_data.mat'); n_taps_per_line = 21; %brick 21pts
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1156/all_data.mat'); n_taps_per_line = 21; %square 21pts


% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1442/all_data.mat'); n_taps_per_line = 6; %flower pt6
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1436/all_data.mat'); n_taps_per_line = 6; %circle pt6
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1426/all_data.mat'); n_taps_per_line = 11; %flower pt11
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1419/all_data.mat'); n_taps_per_line = 11; %circle pt11
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1406/all_data.mat'); n_taps_per_line = 21; %flower pt21
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1357/all_data.mat'); n_taps_per_line = 21; %circle pt21

% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1406/all_data.mat'); n_taps_per_line = 21; %flower step5
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-14_1116/all_data.mat'); n_taps_per_line = 21; %flower step10
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-15_1357/all_data.mat'); n_taps_per_line = 21; %circle step5
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-14_1110/all_data.mat'); n_taps_per_line = 21; %circle step10
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-14_1427/all_data.mat'); n_taps_per_line = 21; %circle step15
% load('/home/lizzie/OneDrive/data/robotcode_first_adapt2019-02-14_1443/all_data.mat'); n_taps_per_line = 21; %circle step20

%% Using GPLVM Only 
% higher_path = '/home/lizzie/OneDrive/data/';
% file_name = '/all_data.mat';

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

% failure tests
% file_name = '/matlab.mat';
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

% full_path = strcat(higher_path, current_folder, file_name);
% load(full_path)

% 252
% clf
dissims=[];
ref = [ex.ref_diffs_norm(ex.ref_diffs_norm_max_ind ,:  ,1) ex.ref_diffs_norm(ex.ref_diffs_norm_max_ind ,:  ,2)];
% ref = [ex.ref_diffs_norm{2}(ex.ref_diffs_norm_max_ind{2} ,:  ,1) ex.ref_diffs_norm{2}(ex.ref_diffs_norm_max_ind{2} ,:  ,2)];

n_ref_taps = 1;

n_lines = size(model.y_gplvm_input_train,1)/n_taps_per_line;%5;

for i = n_ref_taps+1:size(model.y_gplvm_input_train,1)
    
    differences = ref - model.y_gplvm_input_train(i,:); 


    diss = norm([differences(:,1:126)'; differences(:,127:end)']);
    dissims =[dissims; diss]; %#ok<AGROW>
end

hold on 
raw_min_dissim=[];
mins=[]
% y_mins=[]
% z_mins=[]
for i = 1:n_lines
    1+(i-1)*21
    i*21
    x = model.x_gplvm_input_train(1+n_ref_taps+(i-1)*n_taps_per_line:i*n_taps_per_line+n_ref_taps,1);
    y = model.x_gplvm_input_train(1+n_ref_taps+(i-1)*n_taps_per_line:i*n_taps_per_line+n_ref_taps,2);
    z = dissims(1+(i-1)*n_taps_per_line:i*n_taps_per_line);
    plot3( x,y,z,'color',[i/n_lines 0 1-(i/n_lines)])
    text(0, y(1) ,min(z)-2, num2str(i) )
    [raw_min_dissim(i), loc(i)] = min(z);
    mins(1,i) = x(loc(i));
    mins(2,i) = y(loc(i));
    mins(3,i) = z(loc(i));
end
mins = sortrows(mins',2)';
plot3(mins(1,:),mins(2,:),mins(3,:),'k:')

xlabel("Estimated displacement / mm")
ylabel("Predicted \mu")
zlabel("Dissimilarity")
title("GPLVM Model")

% view([-1,0.5,0.2])
view([-1,1.5,2])




daspect([1 0.1 1])
grid on

% print('-dpng', ['/home/lizzie/OneDrive/matlab-figs/', current_folder,'MODEL.png']);
% savefig(['/home/lizzie/OneDrive/matlab-figs/', current_folder,'MODEL.fig']);

