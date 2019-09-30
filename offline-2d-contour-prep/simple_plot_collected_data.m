clearvars

load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c01_01.mat')
all_data{1}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c02_01.mat')
all_data{2}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c03_01.mat')
all_data{3}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c04_01.mat')
all_data{4}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c05_01.mat')
all_data{5}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c06_01.mat')
all_data{6}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c07_01.mat')
all_data{7}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c08_01.mat')
all_data{8}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c09_01.mat')
all_data{9}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c10_01.mat')
all_data{10}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c11_01.mat')
all_data{11}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c12_01.mat')
all_data{12}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c13_01.mat')
all_data{13}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c14_01.mat')
all_data{14}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c15_01.mat')
all_data{15}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c16_01.mat')
all_data{16}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c17_01.mat')
all_data{17}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c18_01.mat')
all_data{18}= fliplr(data);
load('/home/lizzie/git/masters-tactile/data/singleRadius2019-01-17_1046/c19_01.mat')
all_data{19}= fliplr(data);

for radius_num = 1:19
    for tap_num = 1:1:21
        clf
        hold on
        % reference blank tap
        plot(all_data{1}{1}(1,:,1),all_data{1}{1}(1,:,2),'+','Color','k')   
        axis([0 550 0 550])
        xlabel(["Tap Num" tap_num "Radius num" radius_num])
        % moving 
        frame = 6;
%         for frame = 1:14
            plot(all_data{radius_num}{tap_num}(frame,:,1),all_data{radius_num}{tap_num}(frame,:,2),'o','Color',[frame*(1/14),0,0])
            pause(0.1)
        
            axis([0 550 0 550])
        
            xlabel(["Tap Num" tap_num "Radius num" radius_num])
%         end
        hold off
        pause(1)
    end
end
        
        
        
%         max(all_data{1}{6}(:,:,:),[],1) - min(all_data{1}{6}(:,:,:),[],1)
        
        
        
        
        
        
        
        