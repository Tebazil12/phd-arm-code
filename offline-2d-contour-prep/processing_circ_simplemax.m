%% Comparison using ... 
%>- max deflection 
% - sum deflection of all frames, euclidean norm between taps
% - convolution to find time of "peak" deflection , ""
% - convolution to find height of conv to use for dissim , ""
% - 
% - diffs from a gaussian, ""
% - pca 

clear all
clearvars
figure(1)
clf

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
dissimss = [];
    nums =[];
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
                        
            [~,max_ind_x]=max(abs(current_tap_data_norm(: ,pin  ,1)));
            [~,max_ind_y]=max(abs(current_tap_data_norm(: ,pin  ,2)));
            
            max_i(1,pin) = max_ind_x;
            max_i(2,pin) = max_ind_y;
        end
        
        average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2)); % want to compare same frame across tap, not different frames for each pin
        
        differences = ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,:) - current_tap_data_norm(average_max_i,:,:); %TODO this is stupid, ref tap will have its own max diff frame that is not the same 
        
%         diss = sqrt(mean(differences(:,:,1))^2 +mean(differences(:,:,2))^2) ;
        diss =norm([differences(:,:,1) ;differences(:,:,2)]);
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'cosine') % NB, change axis scale to see graphs properly
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'chebychev')
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'euclidean')
%         diss = pdist([differences(:,:,1);differences(:,:,2)], 'cosine') * norm([differences(:,:,1);differences(:,:,2)]);

        num = [num (tap_num)-21];
%         num = [num 2*(tap_num)-12];
        dissims =[dissims diss];
    end
    nums= [nums; num];
    dissimss =[dissimss; dissims];
end
figure(1)
clf
    
%     subplot(5,4,angle_num+1)
%     title(["Radius angle: " ((angle_num-1)*20-160)])
%     hold on
%     %scatter(num, dissims)
%     plot(num, dissims)
%     xlabel("Displacemt / mm")
%     ylabel("dissim")
%     axis([-20 10 0 75])
% %     axis([-20 10 0 15])
% %     axis([-20 10 0 pi/2])
% 
%     hold off
%     
%     subplot(5,4,1)
subplot(2,2,4)
%     title("Whole")
    hold on
    %scatter(num, dissims)
    colour = [1-(angle_num/18) 0 angle_num/18];
    %     plot3(nums', dissimss',1:1:18)%, 'Color',colour)
    surf(nums',ones(31,18).*[-160:20:180], dissimss')%, 'Color',colour)

    xlabel("Displacemt (mm)")
    ylabel("Angle (^o)")
    zlabel("Dissimilarity")
    axis([-20 10 -160 180])
    view([-1,-1,0.2])
    grid on
    grid minor

subplot(2,2,2)
%     title("Above")
    hold on
    %scatter(num, dissims)
    colour = [1-(angle_num/18) 0 angle_num/18];
    %     plot3(nums', dissimss',1:1:18)%, 'Color',colour)
    surf(nums',ones(31,18).*[-160:20:180], dissimss')%, 'Color',colour)

    xlabel("Displacemt (mm)")
    ylabel("Angle (^o)")
    zlabel("Dissimilarity")
    axis([-20 10 -160 180])

    view([270 90]);
    % view([1,-1,-1])
    
subplot(2,2,1)
    plot(nums', dissimss')
    xlabel("Displacemt (mm)")
    ylabel("Dissimilarity")
    % end
    grid on
    grid minor
    
subplot(2,2,3)
%     title("Whole")
    hold on
    %scatter(num, dissims)
    colour = [1-(angle_num/18) 0 angle_num/18];
    %     plot3(nums', dissimss',1:1:18)%, 'Color',colour)
    surf(nums',ones(31,18).*[-160:20:180], dissimss')%, 'Color',colour)

    xlabel("Displacemt (mm)")
    ylabel("Angle (^o)")
    zlabel("Dissimilarity")
    axis([-20 10 -160 180])
    view([-1,0,0])
    grid on
    grid minor    
    
hold off