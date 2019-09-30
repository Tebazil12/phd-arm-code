%% Comparison using ... 
% - sum deflection of all frames, euclidean norm between taps
% - convolution to find time of "peak" deflection , ""
%>- convolution to find height of conv to use for dissim , ""
% - 
% - diffs from a gaussian, ""
% - pca 

clf 
clearvars

%% Load data
load('/home/lizzie/git/masters-tactile/data/wholeCircleRadii2018-10-22_1615/c180_01.mat') % angle changes wrt workframe

% Define tap to be most similar to (sum of all frames)
ref_tap = data{1,21}(:,:,:); % 21st tap in experiment is probably centered on edge (0mm disp)% all the pins so can do conv with each tap

% load('/home/lizzie/git/masters-tactile/data/wholeCircleRadii2018-10-16_1145/c01_01.mat') % angle fixed wrt workframe
% ref_tap = data{1,6}(:,:,:);

data; % data{1,tapnum}(frame ,pin  ,xory)
ref_tap(:,:,1); % for all x data
ref_tap(:,:,2); % for all y data

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,a] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([a(:,:,1) a(:,:,2)]));
ref_diffs_norm_max_ind_from_end = ref_diffs_norm_max_ind -  size(ref_diffs_norm,1)

%% compare this to all taps

for angle_num = 1:1:18
    dissims = [];
    num =[];
    %differences =zeros(1,size(data{1,actual_index},2),size(data{1,actual_index},3));
    for tap_num = 1:1:31
        actual_index = (31*(angle_num-1))+tap_num;
%     for tap_num = 1:1:11
%         actual_index = (11*(angle_num-1))+tap_num;
        current_tap_data_norm = data{1,actual_index}(: ,:  ,:) - data{1,actual_index}(1 ,:  ,:);
        
        all_max_x = zeros(1,127);
        all_max_y = zeros(1,127);
        for pin = 1:127
%             conv_data_x = conv(ref_tap(: ,pin  ,1),data{1,actual_index}(: ,pin  ,1), 'same');
%             conv_data_y = conv(ref_tap(: ,pin  ,2),data{1,actual_index}(: ,pin  ,2), 'same');

%             ref_diff_x = ref_tap(: ,pin  ,1) - ref_tap(1 ,pin  ,1);
%             ref_diff_y = ref_tap(: ,pin  ,2) - ref_tap(1 ,pin  ,2);
%             
%             data_diff_x = data{1,actual_index}(: ,pin  ,1)-data{1,actual_index}(1 ,pin  ,1);
%             data_diff_y = data{1,actual_index}(: ,pin  ,2)-data{1,actual_index}(1 ,pin  ,2);
%             
%             conv_data_x = conv( ref_diff_x, data_diff_x, 'same');
%             conv_data_y = conv( ref_diff_y, data_diff_y,  'same');


            conv_data_x = conv(fliplr(ref_diffs_norm(: ,pin  ,1)), current_tap_data_norm(: ,pin  ,1), 'full'); % fliplr because conv compares the reflection %TODO see if Cross-correlation (xcorr) gives same result
            conv_data_y = conv(fliplr(ref_diffs_norm(: ,pin  ,2)), current_tap_data_norm(: ,pin  ,2), 'full');
%  
%             conv_data_x = xcorr((ref_diffs_norm(: ,pin  ,1)), current_tap_data_norm(: ,pin  ,1)); %TODO come back to this
%             conv_data_y = xcorr((ref_diffs_norm(: ,pin  ,2)), current_tap_data_norm(: ,pin  ,2));

            [~,max_ind_x]=max(abs(conv_data_x))
            [~,max_ind_y]=max(abs(conv_data_y))
            
            max_val_x = conv_data_x(max_ind_x + ref_diffs_norm_max_ind_from_end );
            max_val_y = conv_data_y(max_ind_y + ref_diffs_norm_max_ind_from_end );

            %average_max_i = round((average_max_i + (max_ind_x + max_ind_y)/2)/2);% TODO maybe use median instead (whacky nums with mean if one anomoly? Might be same with median though...)
            all_max_x(pin) = max_val_x/length(conv_data_x);
            all_max_y(pin) = max_val_y/length(conv_data_x);
        end
        %average_max_i
        
        %differences = ref_tap(average_max_i,:,:) - data{1,actual_index}(average_max_i,:,:);

        %diss =norm([differences(:,:,1);differences(:,:,2)]);
        all_max_x;
        all_max_y;
        diss =norm([all_max_x ;all_max_y ]);
%         diss = sqrt(mean(all_max_y)^2 +mean(all_max_x)^2) ;
%         diss = pdist2(all_max_x,all_max_y, 'euclidean')
        num = [num (tap_num)-21];
%         num = [num 2*(tap_num)-12];
        dissims =[dissims diss];
    end
    
    figure(1)
    subplot(5,4,angle_num+1)
    title(["Radius angle: " ((angle_num-1)*20-160)])
    hold on
    %scatter(num, dissims)
    plot(num, dissims)
%     plot([-20 10],[100 100])
%     plot([-20 10],[120 100])
%     plot([0 10],[110 20])
    plot([0 0],[0 120])
    plot([0 0],[0 6])
    xlabel("Displacemt / mm")
    ylabel("dissim")
    axis([-20 10 0 120])
    hold off
    
    subplot(5,4,1)
    title("All angles")
    hold on
    %scatter(num, dissims)
    plot(num, dissims)
    xlabel("Displacemt / mm")
    ylabel("dissim")
    %axis([-20 10 1948000 1958000])
    hold off
end
