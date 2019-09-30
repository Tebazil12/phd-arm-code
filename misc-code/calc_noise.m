%% Load data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_01.mat')
all_data{1}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_02.mat')
all_data{2}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_03.mat')
all_data{3}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_04.mat')
all_data{4}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_05.mat')
all_data{5}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_06.mat')
all_data{6}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_07.mat')
all_data{7}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_08.mat')
all_data{8}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_09.mat')
all_data{9}= data
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_10.mat')
all_data{10}= data

i =[]
for radii = 1:10
    for tap = 1:31
        i = [i; (var(all_data{radii}{1,tap}(:,:,1)))];
        i = [i; (var(all_data{radii}{1,tap}(:,:,2)))];
    end
end
mean(mean(i))