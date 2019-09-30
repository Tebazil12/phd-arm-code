clf 
clearvars

%% Load data
load('/home/lizzie/git/masters-tactile/data/wholeCircleRadii2018-10-22_1615/c180_01.mat')

data; % data{1,tapnum}(frame ,pin  ,xory)

%% Comparison using ... 
% - sum deflection of all frames, euclidean norm between taps
% - convolution to find time of "peak" deflection , ""
% - convolution to find height of conv to use for dissim , ""
% - 
% - diffs from a gaussian, ""
% - pca 

% Define tap to be most similar to (sum of all frames)
ref_tap = data{1,21}(:,:,:); % 21st tap in experiment is probably centered on edge (0mm disp)
% all the pins so can do conv with each tap

ref_tap(:,:,1); % for all x data
ref_tap(:,:,2); % for all y data

%% compare this to all taps
%ref_frame = 6;


% NOTE this method may not work as some taps are U shaped not just n shaped
% this messes with the height of the conv., and maybe the max point?
% Investigate/ prove!


for angle_num = 1:17 %17
    dissims = [];
    num =[];
    %differences =zeros(1,size(data{1,actual_index},2),size(data{1,actual_index},3));
    for tap_num = 1:31 %31
        actual_index = (31*(angle_num-1))+tap_num;
        %average_max_i =0;
        all_max_x = zeros(1,127);
        all_max_y = zeros(1,127);
        for pin = 1:127 %127 %TODO note that this will error if a pin is not detected...
            ref_diff_x = ref_tap(: ,pin  ,1) - ref_tap(1 ,pin  ,1);
            ref_diff_y = ref_tap(: ,pin  ,2) - ref_tap(1 ,pin  ,2);
            
            data_diff_x = data{1,actual_index}(: ,pin  ,1)-data{1,actual_index}(1 ,pin  ,1);
            data_diff_y = data{1,actual_index}(: ,pin  ,2)-data{1,actual_index}(1 ,pin  ,2);
            
            conv_data_x = conv(ref_diff_x, data_diff_x, 'same');
            conv_data_y = conv(ref_diff_y, data_diff_y, 'same');
            
            [max_val_x,max_ind_x]=max(conv_data_x);
            [max_val_y,max_ind_y]=max(conv_data_y);
            
%             figure(tap_num-14)
%             subplot(1,2,1)
%             title(["Pin: " pin])
%             hold on
%             plot(1:length(conv_data_x), conv_data_x)
%             hold off
%             
%             subplot(1,2,2)
%             title(["Pin: " pin])
%             hold on
%             plot(1:length(conv_data_y), conv_data_y)
%             hold off

            %average_max_i = round((average_max_i + (max_ind_x + max_ind_y)/2)/2);% TODO maybe use median instead (whacky nums with mean if one anomoly? Might be same with median though...)
            all_max_x(pin) = max_val_x;
            all_max_y(pin) = max_val_y;
        end
        %average_max_i
        
        %differences = ref_tap(average_max_i,:,:) - data{1,actual_index}(average_max_i,:,:);

        %diss =norm([differences(:,:,1);differences(:,:,2)]);
        all_max_x
        all_max_y
        %diss =norm([all_max_x ,all_max_y ]);
        diss = sqrt(mean(all_max_y)^2 +mean(all_max_x)^2) ;
        num = [num (tap_num)-21];
        dissims =[dissims diss];
    end
    
    figure(2)
    subplot(5,4,angle_num+1)
    title(["Tactip angle: " ((angle_num-1)*20-160)])
    hold on
    %scatter(num, dissims)
    plot(num, dissims)
    xlabel("Displacemt / mm")
    ylabel("dissim")
    %axis([-20 10 1948000 1958000])
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
