%% Comparison using ... 
%>- max deflection 
% - sum deflection of all frames, euclidean norm between taps
% - convolution to find time of "peak" deflection , ""
% - convolution to find height of conv to use for dissim , ""
% - 
% - diffs from a gaussian, ""
% - pca 

clf 
clearvars

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

% Define tap to be most similar to (sum of all frames)
ref_tap = all_data{1}{1,11}(:,:,:); % 11th tap (cuz reverse collection order) in experiment is probably centered on edge (0mm disp)% all the pins so can do conv with each tap


data; % data{1,tapnum}(frame ,pin  ,xory)
ref_tap(:,:,1); % for all x data
ref_tap(:,:,2); % for all y data

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));
% ref_diffs_norm_max_ind_from_end = ref_diffs_norm_max_ind -  size(ref_diffs_norm,1);

%% compare this to all taps

for repeat_index = 1:10
    dissims = [];
    num =[];

    for tap_num = 1:31
        current_tap_data_norm = all_data{repeat_index}{1,tap_num}(: ,:  ,:)...
                                - all_data{repeat_index}{1,tap_num}(1 ,:  ,:);

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

        num = [num (31-(tap_num))-20];
%         num = [num 2*(tap_num)-12];
        dissims =[dissims diss];
    end
    
    figure(1)
%     subplot(5,3,repeat_index+1)
%     title(["Radius angle: " repeat_index])
%     hold on
%     scatter(num, dissims,'.')
% %     plot(num, dissims)
%     xlabel("Displacemt / mm")
%     ylabel("dissim")
%     axis([-20 10 0 40])
% %     axis([-20 10 0 15])
% %     axis([-20 10 0 pi/2])
% 
%     hold off
%     
%     subplot(5,3,1)
    title("All angles")
    hold on
%     scatter(num, dissims,'+')
    plot(num, dissims)
    xlabel("Displacemt / mm")
    ylabel("dissim")
    axis([-20 10 0 40])
    hold off
    
    all_dissims(repeat_index,:) = dissims;
end
all_dissims
var(all_dissims)
