%% Comparison using ... 
% - sum deflection of all frames, euclidean norm between taps
%>- convolution to find time of "peak" deflection , ""
% - convolution to find height of conv to use for dissim , ""
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
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));
ref_diffs_norm_max_ind_from_end = ref_diffs_norm_max_ind -  size(ref_diffs_norm,1);

%% compare this to all taps

for angle_num = 1:1:18
    dissims = [];
    num =[];

    for tap_num = 1:1:31
        actual_index = (31*(angle_num-1))+tap_num;
%     for tap_num = 1:1:11
%         actual_index = (11*(angle_num-1))+tap_num;
        current_tap_data_norm = data{1,actual_index}(: ,:  ,:) - data{1,actual_index}(1 ,:  ,:);

        max_i = zeros(1, 127);
        for pin = 1:127
            
            conv_data_x = conv(fliplr(ref_diffs_norm(: ,pin  ,1)), current_tap_data_norm(: ,pin  ,1), 'full'); % fliplr because conv compares the reflection %TODO see if Cross-correlation (xcorr) gives same result
            conv_data_y = conv(fliplr(ref_diffs_norm(: ,pin  ,2)), current_tap_data_norm(: ,pin  ,2), 'full');
            
%             conv_data_x = xcorr((ref_diffs_norm(: ,pin  ,1)), current_tap_data_norm(: ,pin  ,1)); %TODO come back to this - indexing may not be the same as for conv
%             conv_data_y = xcorr((ref_diffs_norm(: ,pin  ,2)), current_tap_data_norm(: ,pin  ,2));
            
            [~,max_ind_x]=max(abs(conv_data_x));
            [~,max_ind_y]=max(abs(conv_data_y));
            
            max_i(1,pin) = max_ind_x + ref_diffs_norm_max_ind_from_end; % becuase conv works off leading edge, not max index
            max_i(2,pin) = max_ind_y + ref_diffs_norm_max_ind_from_end;
        end
        
        average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2));
        
        differences = ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,:) - current_tap_data_norm(average_max_i,:,:);
        
%         diss = sqrt(mean(differences(:,:,1))^2 +mean(differences(:,:,2))^2) ;
%         diss =norm([differences(:,:,1) ;differences(:,:,2)]);
        diss = pdist([differences(:,:,1); differences(:,:,2)], 'cosine') % NB, change axis scale to see graphs properly
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'chebychev')
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'euclidean')
%         diss = pdist([differences(:,:,1);differences(:,:,2)], 'cosine') * norm([differences(:,:,1);differences(:,:,2)]);

        num = [num (tap_num)-21];
%         num = [num 2*(tap_num)-12];
        dissims =[dissims diss];
    end
    
    figure(1)
    subplot(5,4,angle_num+1)
    title(["Radius angle: " ((angle_num-1)*20-160)])
    hold on
%     scatter(num, dissims,'+')
    plot(num, dissims)
    xlabel("Displacemt / mm")
    ylabel("dissim")
%     axis([-20 10 0 75])
%     axis([-20 10 0 15])
    axis([-20 10 0 pi/2])

    hold off
    
    subplot(5,4,1)
    title("All angles")
    hold on
    %scatter(num, dissims)
    plot(num, dissims)
    xlabel("Displacemt / mm")
    ylabel("dissim")
    %axis([-20 10 0 60])
    hold off
end
