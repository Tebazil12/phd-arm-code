%% Load data from repeated single radii exp.
% x will be displacement (later also angle/depth)
% y will be normalized tap at max disp (raw data minus first frame, max
% disps of those, diff with reference tap).

clearvars
figure(1)
clf
figure(2)
clf
figure(3)
clf
figure(4)
clf
clear all

% load('/home/lizzie/git/masters-tactile/data/wholeCircleRadii2018-10-22_1615/c180_01.mat') % whole circ, moving angle wrt workframe
% 
% % load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_01.mat')
% for radii_num = 1:18
%     for num = 1:31
%         actual_index = (31*(radii_num-1))+num;
%         all_data{radii_num}{1,num}= data{1,actual_index};
%     end
% end
% all_data{1}
% max_num= 18;

load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_01.mat')
all_data{1}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_02.mat')
all_data{2}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_03.mat')
all_data{3}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_04.mat')
all_data{4}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_05.mat')
all_data{5}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_06.mat')
all_data{6}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_07.mat')
all_data{7}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_08.mat')
all_data{8}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_09.mat')
all_data{9}= fliplr(data)
load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_10.mat')
all_data{10}= fliplr(data)
max_num = 10;

x_real(:,1) = [-20:10]';
x_real_test = [-20:10]';

%% Define reference tap & stuff
ref_tap = all_data{1}{1,21};%(:,:,:); 

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));

%% Filter raw data (normalize, find max frame)
sigma_n_y = 1.94;%par(3); %TODO this is wrong, implement higher up
sigma_n_diss = 1.94;

[dissims1, y_train1,x_diffs1,y_diffs1] = process_taps(all_data{1},ref_diffs_norm,ref_diffs_norm_max_ind,sigma_n_diss, x_real(:,1))
[dissims2, y_train2,x_diffs2,y_diffs2] = process_taps(all_data{2},ref_diffs_norm,ref_diffs_norm_max_ind,sigma_n_diss, x_real(:,1))

%% Optimize hyper-params for "training"

init_hyper_pars = [7 2];


y_gplvm_input = [[x_diffs1';x_diffs2'] [y_diffs1';y_diffs2']+0.00000001];
x_gplvm_input = [ [x_real(:,1);x_real(:,1)] [zeros(size(x_real(:,1)));10*ones(size(x_real(:,1)))] ];

[par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(opt_pars(1), opt_pars(2), sigma_n_y,...
                                                          y_gplvm_input ,x_gplvm_input),...
                            init_hyper_pars)

if flag ~= 1
    warning("fminsearch was not happy")
    flag
end 

if round(par(1),1) == 0 || round(par(2),1) == 0
    warning("A hyper-parameter is zero! Bad fit")
end

sigma_f = par(1);
l = par(2);

%% "Test" stuff
fprintf('Calculating predictions\n')
n_bad_flags = 0;
for set_num = 2:max_num
    [dissims_test, y_test,x_diffs_test,y_diffs_test] = process_taps(all_data{set_num},ref_diffs_norm,ref_diffs_norm_max_ind,sigma_n_diss, x_real(:,1));
    
%     x_shift = actual_x_min + x
    
    
    x_real(:,set_num) = x_real(:,1);%+x_shift-actual_x_min;
    for test_tap_num = 1:31
        init_latent_vars = [0];

        [par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f, l, sigma_n_y,...
                                                                  [y_gplvm_input; [x_diffs_test(test_tap_num) y_diffs_test(test_tap_num)]] , [x_gplvm_input; test_tap_num+0.001 opt_pars(1) ]),...
                                    init_latent_vars , optimoptions('fminunc','Display','off'));

        if flag ~= 1
            n_bad_flags = n_bad_flags +1;
        end 

        new_x(set_num-1, test_tap_num,:) = par;
        dissims_tests(set_num-1, test_tap_num) = dissims_test(test_tap_num);
        x_error(set_num-1, test_tap_num,:) = par - (test_tap_num - 20);
        
        fprintf('.');
    end
    fprintf('\n')
end

new_x
dissims_tests
x_error
max_x_error = max(abs(x_error))

n_bad_flags
%% Plot everything

figure(1)
title("All angles")
hold on
xlabel("Displacemt / mm")
ylabel("dissim")
% axis([-20 10 0 40])

% % plot Stdev band
% fill([x_stars, fliplr(x_stars)],...
%      [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
%      [1 1 0.8],...
%      'EdgeColor','none')
% 
% % plot predictions from interpolating
% plot(x_stars,y_star, 'r') %TODO how does this even work?!

% plot actual data
% plot(x_real(:,1), dissims, 'k')
% scatter(x_real, dissims, 'k','+')

plot(x_real(:,2:end), dissims_tests','r')
% scatter(x_real, dissims_test,'r','+')
% 
% plot(10:-1:-20, dissims_test,'r')
% scatter(10:-1:-20, dissims_test,'r','+')

% plot predicted data
plot(new_x(:,:,1),dissims_tests, '+') 
% scatter(new_x,dissims_tests, 'b') 

plot([(10+1)-test_tap_num (10+1)-test_tap_num],[0 30],'b')



grid on
grid minor

hold off

figure(2)
% plot((31:-1:1)',x_error','o')
pos_x_err_bool = (x_error >= 0);
pos_x_err = pos_x_err_bool .* x_error;
neg_x_err_bool = (x_error < 0);
neg_x_err = neg_x_err_bool .* x_error;

errorbar(new_x',dissims_tests',pos_x_err',neg_x_err','o','horizontal')

grid on
grid minor

figure(3)

subplot(1,3,1)
hold on
% plot3(x_real(:,2:end)',new_x(:,:,1),new_x(:,:,2),'+')
plot([-20 10],[-20 10],'k')
xlabel("Real Displacemt / mm")
ylabel("Predicted Displacemt / mm")
zlabel("mu")
hold off
grid on
grid minor

subplot(1,3,2)
hold on
plot(x_real(:,2:end)',new_x(:,:,1),'+')
plot([-20 10],[-20 10],'k')
xlabel("Real Displacemt / mm")
ylabel("Predicted Displacemt / mm")
grid on
grid minor
hold off

subplot(1,3,3)
hold on
plot(x_real(:,2:end),new_x(:,:,1)','+')
plot(x_real(:,2:end),new_x(:,:,1)')
plot(x_gplvm_input(:,1),x_gplvm_input(:,2),'o')
xlabel("Real Displacemt / mm")
ylabel("mu / mm")
grid on
grid minor
hold off

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------- FUNCTIONS ---------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dissims, y_for_real, x_diffs, y_diffs] = process_taps(radii_data, ref_diffs_norm, ref_diffs_norm_max_ind, sigma_n, x_matrix)
    % Return modified y data, dissimilarity data and the minimum x point
    % for a single radius of data (intended to be a single radius).
    
    y_for_real = [];
    dissims = [];
    x_diffs =[];
    y_diffs =[];
    for tap_num = 1:length(radii_data)
        current_tap_data_norm = radii_data{1,tap_num}(: ,:  ,:)...
                                - radii_data{1,tap_num}(1 ,:  ,:);

        max_i = zeros(1, 127);
        for pin = 1:127

            [~,max_ind_x]=max(abs(current_tap_data_norm(: ,pin  ,1)));
            [~,max_ind_y]=max(abs(current_tap_data_norm(: ,pin  ,2)));

            max_i(1,pin) = max_ind_x;
            max_i(2,pin) = max_ind_y;
        end

        average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2)); % want to compare same frame across tap, not different frames for each pin

        differences = ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,:) ...
                      - current_tap_data_norm(average_max_i,:,:); 
%         differences = current_tap_data_norm(average_max_i,:,:); 
                  
                  
        y_for_real = [y_for_real; differences(:,:,1) differences(:,:,2)];
        diss = norm([differences(:,:,1)' ;differences(:,:,2)']);
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'euclidean')
        
        dissims =[dissims diss];
        x_diffs = [x_diffs sum(differences(:,:,1))];
        y_diffs = [y_diffs sum(differences(:,:,2))];
        
    end
    [x_diffs' y_diffs'];
    
%     [par, fval, flag] = fminunc(@(mypar)gplvm_max_log_like(mypar(1), mypar(2), sigma_n, [x_diffs' y_diffs'] , x_matrix), [0.5 2] )
%     
%     sigma_f = par(1);
%     l = par(2);
% 
% 
%     k_cap = calc_k_cap(x_matrix, sigma_f,l, sigma_n)
% 
%     i = 1;
%     for x_star = -20:0.1:10
%         if sum(-20:10 == x_star) == 0
%             x_stars(i) = x_star;
% 
%             % setup covariance matrix stuff
%             k_star      = calc_k_star(x_star, x_matrix, sigma_f,l, sigma_n);
%             k_star_star = calc_covar_ij(x_star, x_star, sigma_f,l, sigma_n);
% 
%             % Estimate y
%             y_star(i,:) = k_star * inv(k_cap) * [x_diffs' y_diffs'];
% 
%             % Estimate variance
%             var_y_star(i) = k_star_star - (k_star * inv(k_cap) * transpose(k_star));
%             if var_y_star(i) < 0.0000
%                 var_y_star(i) =0; % otherwise -0.0000 causes errors with sqrt()
%             end
% 
%             i = i+1;
%         end
%     end
%     figure(2)
% %     clf
%     hold on
% %     fill([x_stars, fliplr(x_stars)],...
% %          [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
% %          [1 1 0.8])
%     
%     plot(x_matrix(:,1), dissims, '+')
%     plot(x_stars, y_star)
%     hold off
% %     [~,x_min_ind] = min(y_star)
% %     x_shift = -x_stars(x_min_ind)
%     
%     figure(4)
%     hold on
%     plot(x_matrix(:,1)+x_shift, dissims, '+')
%     plot(x_matrix(:,1)+x_shift, dissims)
%     grid on
%     grid minor
%     axis([-20 10 0 75])
%     hold off
% %     y_for_real
end