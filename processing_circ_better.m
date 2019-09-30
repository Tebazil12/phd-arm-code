clf 
clearvars

%% Load data
load('/home/lizzie/git/masters-tactile/data/wholeCircleRadii2018-10-22_1615/c180_01.mat')

data; % data{1,tapnum}(frame ,pin  ,xory)

%% Comparison using sum of all frames 

% Define tap to be most similar to (sum of all frames)
% ref_tap = sum(data{1,21}(:,:,:),1) / size(data{1,21},1); % 21st tap in experiment is probably centered on edge (0mm disp)
ref_tap = sum(data{1,21}(:,:,:) - data{1,21}(1,:,:),1) / size(data{1,21},1); 
% normalized as sometimes 14 frames, sometimes 15

ref_tap(:,:,1); % for all x data
ref_tap(:,:,2); % for all y data

%% compare this to all taps
ref_frame = 6;
for angle_num = 1:1:17
    dissims = [];
    num =[];
    %differences =zeros(1,size(data{1,actual_index},2),size(data{1,actual_index},3));
    for tap_num = 1:1:31
        actual_index = (31*(angle_num-1))+tap_num;
        
%         new_tap = sum(data{1,actual_index}(:,:,:),1) / size(data{1,actual_index},1);
        new_tap = sum(data{1,actual_index}(:,:,:)-data{1,actual_index}(1,:,:),1) / size(data{1,actual_index},1);


        differences = ref_tap - new_tap;

        diss =norm([differences(:,:,1);differences(:,:,2)]);
        num = [num (tap_num)-21];
        dissims =[dissims diss];
    end
    
    figure(1)
    subplot(5,4,angle_num+1)
    title(["Tactip angle: " ((angle_num-1)*20-160)])
    hold on
    %scatter(num, dissims)
    plot(num, dissims)
    xlabel("Displacemt / mm")
    ylabel("dissim")
    axis([-20 10 0 21])
    hold off
    
    subplot(5,4,1)
    title("All angles")
    hold on
    %scatter(num, dissims)
    plot(num, dissims)
    xlabel("Displacemt / mm")
    ylabel("dissim")
    axis([-20 10 0 21])
    hold off
end
