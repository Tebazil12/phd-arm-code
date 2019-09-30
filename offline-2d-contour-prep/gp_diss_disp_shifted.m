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
% data1 = data

load('/home/lizzie/git/masters-tactile/data/wholeCircleRadii2018-10-16_1145/c01_01.mat') % whole circ, fixed angle wrt workframe
data;

% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_01.mat')
for radii_num = 1:18
    for num = 1:11
        actual_index = (11*(radii_num-1))+num;
        all_data{radii_num}{1,num}= data{1,actual_index};
    end
end
all_data{1};
max_num= 18;

TRIM_DATA = true;
TRAIN_MIN_DISP = -3;



% % Repettions of exact same radius
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_01.mat')
% all_data{1}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_02.mat')
% all_data{2}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_03.mat')
% all_data{3}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_04.mat')
% all_data{4}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_05.mat')
% all_data{5}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_06.mat')
% all_data{6}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_07.mat')
% all_data{7}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_08.mat')
% all_data{8}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_09.mat')
% all_data{9}= fliplr(data)
% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_10.mat')
% all_data{10}= fliplr(data)
% max_num = 10;

% For all, max = 11, min = 1
min_index_test = 1;
max_index_test = 8;
min_i_train = 1;
max_i_train = 11;

x_real(:,1) = [-10:2:10]';
x_real_test = [-10:2:10]';

%% Define reference tap & stuff
ref_tap = all_data{1}{1,6};%(:,:,:); 

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));

%% Filter raw data (normalize, find max frame)
sigma_n_y = 1.94;
sigma_n_diss = 1.94;

[dissims1, y_train1, x_diffs1, y_diffs1] = process_taps(all_data{1},...
                                                        ref_diffs_norm,...
                                                        ref_diffs_norm_max_ind,...
                                                        sigma_n_diss,...
                                                        x_real(:,1),...
                                                        ref_tap);
x_min1  = radius_diss_shift(dissims1, x_real(:,1), sigma_n_diss);
if x_min1 ~= 0
    warning("Reference tap is not the min at disp 0")
end

[dissims2, y_train2, x_diffs2, y_diffs2] = process_taps(all_data{5},...
                                                        ref_diffs_norm,...
                                                        ref_diffs_norm_max_ind,...
                                                        sigma_n_diss,... 
                                                        x_real(:,1),...
                                                        ref_tap);
x_min2  = radius_diss_shift(dissims2, x_real(:,1), sigma_n_diss);
x_real(:,2) = x_real(:,1) + x_min2 ; % so all minima are aligned
if TRIM_DATA
    x_real(:,2) = (x_real(:,2) >TRAIN_MIN_DISP).* x_real(:,2) + (x_real(:,2)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
end

[dissims3, y_train3, x_diffs3, y_diffs3] = process_taps(all_data{10},...
                                                        ref_diffs_norm,...
                                                        ref_diffs_norm_max_ind,...
                                                        sigma_n_diss,...
                                                        x_real(:,1),...
                                                        ref_tap);
x_min3  = radius_diss_shift(dissims3, x_real(:,1), sigma_n_diss);
x_real(:,3) = x_real(:,1) + x_min3 ; % so all minima are aligned
if TRIM_DATA
    x_real(:,3) = (x_real(:,3) >TRAIN_MIN_DISP).* x_real(:,3) + (x_real(:,3)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
end

[dissims4, y_train4, x_diffs4, y_diffs4] = process_taps(all_data{14},...
                                                        ref_diffs_norm,...
                                                        ref_diffs_norm_max_ind,...
                                                        sigma_n_diss,...
                                                        x_real(:,1),...
                                                        ref_tap);
x_min4  = radius_diss_shift(dissims4, x_real(:,1), sigma_n_diss);
 x_real(:,4) = x_real(:,1) + x_min4 ; % so all minima are aligned
if TRIM_DATA
   x_real(:,4) = (x_real(:,4) >TRAIN_MIN_DISP).* x_real(:,4) + (x_real(:,4)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
end

%% Optimize hyper-params for "training"

init_hyper_pars = [7 2 2];
size_x2 = size(x_real(min_i_train:max_i_train,1));

y_gplvm_input = [y_train1(min_i_train:max_i_train,:);...
                 y_train2(min_i_train:max_i_train,:);...
                 y_train3(min_i_train:max_i_train,:);...
                 y_train4(min_i_train:max_i_train,:)];

x_gplvm_input = [ [x_real(min_i_train:max_i_train,1);...
                   x_real(min_i_train:max_i_train,2);...
                   x_real(min_i_train:max_i_train,3);...
                   x_real(min_i_train:max_i_train,4)] ...
                  [zeros(size_x2);...
                   1*ones(size_x2);...
                   2*ones(size_x2);...
                   3*ones(size_x2)] ];

%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
[par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(opt_pars(1), ...
                                                          [opt_pars(2) opt_pars(3)], ...
                                                          sigma_n_y,...
                                                          y_gplvm_input ,x_gplvm_input),...
                            init_hyper_pars,...
                            optimoptions('fminunc','Display','off'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if flag ~= 1
    warning("fminsearch was not happy")
    flag
end 

if round(par(1),1) == 0 || round(par(2),1) == 0
    warning("A hyper-parameter is zero! Bad fit")
end

sigma_f = par(1)
l_disp = par(2)
l_mu = par(3)

test_plot(x_gplvm_input, y_gplvm_input, sigma_f, l_disp, l_mu, sigma_n_y,ref_tap,ref_diffs_norm,ref_diffs_norm_max_ind)

%% "Test" stuff
fprintf('Calculating predictions\n')
n_bad_flags = 0;
n_flags = 0;
for set_num = 2:max_num
    [dissims_test, y_test, x_diffs_test, y_diffs_test] = process_taps(all_data{set_num},...
                                                                      ref_diffs_norm,...
                                                                      ref_diffs_norm_max_ind,...
                                                                      sigma_n_diss,...
                                                                      x_real(:,1),...
                                                                      ref_tap);
    
    x_min  = radius_diss_shift(dissims_test, x_real(:,1), sigma_n_diss);
    
    x_real(:,set_num) = x_real(:,1) + x_min;
    if TRIM_DATA
        x_real(:,set_num) = (x_real(:,set_num) >=TRAIN_MIN_DISP).* x_real(:,set_num)...
                            + (x_real(:,set_num)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
    end
        
    for test_tap_num = min_index_test:max_index_test
        init_latent_vars = [0];
%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f,...
                                                                  [l_disp l_mu],...
                                                                  sigma_n_y,...
                                                                  [y_gplvm_input; y_test(test_tap_num,:)],...
                                                                  [x_gplvm_input; test_tap_num opt_pars(1) ]),...
                                    init_latent_vars,...
                                    optimoptions('fminunc','Display','off'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if flag ~= 1
            n_bad_flags = n_bad_flags +1;
        end 

        new_x(set_num, test_tap_num,:) = par;
        dissims_tests(set_num, test_tap_num) = dissims_test(test_tap_num);
        x_error(set_num, test_tap_num,:) = par - (test_tap_num - 10);
        
        n_flags = n_flags +1;
        
        fprintf('.');
    end
    fprintf('\n')
end

% Print things out
new_x
dissims_tests
x_error
max_x_error = max(abs(x_error))

n_bad_flags
n_flags

%% Plot everything

% figure(1)
% title("All angles")
% hold on
% xlabel("Displacemt / mm")
% ylabel("dissim")
% % axis([-20 10 0 40])
% 
% % % plot Stdev band
% % fill([x_stars, fliplr(x_stars)],...
% %      [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
% %      [1 1 0.8],...
% %      'EdgeColor','none')
% % 
% % % plot predictions from interpolating
% % plot(x_stars,y_star, 'r') %TODO how does this even work?!
% 
% % plot actual data
% plot(x_real(:,1), dissims, 'k')
% % scatter(x_real, dissims, 'k','+')
% 
% plot(x_real(:,2:end), dissims_tests','r')
% % scatter(x_real, dissims_test,'r','+')
% % 
% % plot(10:-1:-20, dissims_test,'r')
% % scatter(10:-1:-20, dissims_test,'r','+')
% 
% % plot predicted data
% plot(new_x(:,:,1),dissims_tests, '+') 
% % scatter(new_x,dissims_tests, 'b') 
% 
% plot([(10+1)-test_tap_num (10+1)-test_tap_num],[0 30],'b')
% 
% 
% 
% grid on
% grid minor
% 
% hold off
% 
% figure(2)
% % plot((31:-1:1)',x_error','o')
% pos_x_err_bool = (x_error >= 0);
% pos_x_err = pos_x_err_bool .* x_error;
% neg_x_err_bool = (x_error < 0);
% neg_x_err = neg_x_err_bool .* x_error;
% 
% errorbar(new_x',dissims_tests',pos_x_err',neg_x_err','o','horizontal')
% 
% grid on
% grid minor

figure(3)

subplot(1,3,1)
hold on
% plot3(x_real(min_index_test:max_index_test,1:end),new_x(:,min_index_test:max_index_test,1)',dissims_tests(:,min_index_test:max_index_test)','+')
plot3(x_real(min_index_test:max_index_test,1:end),new_x(:,min_index_test:max_index_test,1)',dissims_tests(:,min_index_test:max_index_test)')
% plot3(x_gplvm_input(:,1),x_gplvm_input(:,2),[dissims1(min_i_train:max_i_train)';dissims2(min_i_train:max_i_train)';dissims3(min_i_train:max_i_train)';dissims4(min_i_train:max_i_train)'], 'o')
% plot3(x_gplvm_input(:,1),x_gplvm_input(:,2),[dissims1(min_i_train:max_i_train)';dissims2(min_i_train:max_i_train)';dissims3(min_i_train:max_i_train)';dissims4(min_i_train:max_i_train)'])

surf([x_gplvm_input(1:size(x_real,1),1)...
      x_gplvm_input(size(x_real,1)+1:2*size(x_real,1),1)...
      x_gplvm_input(2*size(x_real,1)+1:3*size(x_real,1),1)...
      x_gplvm_input(3*size(x_real,1)+1:4*size(x_real,1),1)],...
     [x_gplvm_input(1:size(x_real,1),2)...
      x_gplvm_input(size(x_real,1)+1:2*size(x_real,1),2)...
      x_gplvm_input(2*size(x_real,1)+1:3*size(x_real,1),2)...
      x_gplvm_input(3*size(x_real,1)+1:4*size(x_real,1),2)],...
     [dissims1(min_i_train:max_i_train)'...
      dissims2(min_i_train:max_i_train)'...
      dissims3(min_i_train:max_i_train)'...
      dissims4(min_i_train:max_i_train)'])%,...
%      'FaceAlpha',0.5)

% plot([-20 10],[-20 10],'k')
xlabel("Real Displacemt / mm")
zlabel("Predicted Displacemt / mm")
ylabel("mu")
hold off
grid on
grid minor
% 
% subplot(1,3,2)
% hold on
% plot(x_real(:,2:end)',new_x(:,:,1),'+')
% plot([-20 10],[-20 10],'k')
% xlabel("Real Displacemt / mm")
% ylabel("Predicted Displacemt / mm")
% grid on
% grid minor
% hold off

subplot(1,3,3)
hold on
plot(x_real(min_index_test:max_index_test,1:end),new_x(:,min_index_test:max_index_test,1)','+')
plot(x_real(min_index_test:max_index_test,1:end),new_x(:,min_index_test:max_index_test,1)')
plot(x_gplvm_input(:,1),x_gplvm_input(:,2),'o')
xlabel("Real Displacemt / mm")
ylabel("mu / mm")
grid on
grid minor
hold off

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------- FUNCTIONS ---------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [dissims, y_for_real, x_diffs, y_diffs] = process_taps(radii_data, ref_diffs_norm, ref_diffs_norm_max_ind, sigma_n, x_matrix, ref_tap)
    % Return modified y data, dissimilarity data and the minimum x point
    % for a single radius of data (intended to be a single radius).
    
    Y_RAW = false;
    Y_NORM = true;
    Y_NORM_DIFF = false;
    
    y_for_real = [];
    dissims = [];
    x_diffs =[];
    y_diffs =[];
    for tap_num = 1:length(radii_data)
        current_tap_data_norm = radii_data{1,tap_num}(: ,:  ,:)...
                                - radii_data{1,tap_num}(1 ,:  ,:);
%     current_tap_data_norm = radii_data{1,tap_num}(: ,:  ,:) - ref_tap(1 ,:  ,:);
    diff_between_noncontacts = ref_tap(1 ,:  ,:) - radii_data{1,tap_num}(1 ,:  ,:); %TODO throw error if too large?
    
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
%         differences = current_tap_data_norm(average_max_i,:,:) - ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,:);
                      
%         differences = current_tap_data_norm(average_max_i,:,:); 
        if Y_RAW 
            y_for_real = [y_for_real;  radii_data{1,tap_num}(average_max_i,:,1)  radii_data{1,tap_num}(average_max_i,:,2)];
        elseif Y_NORM
            y_for_real = [y_for_real; current_tap_data_norm(average_max_i,:,1) current_tap_data_norm(average_max_i,:,2)];
        elseif Y_NORM_DIFF
            y_for_real = [y_for_real; differences(:,:,1) differences(:,:,2)];
        else
            error("Y not chosen")
        end
        diss = norm([differences(:,:,1)' ;differences(:,:,2)']);
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'euclidean')
        
        dissims =[dissims diss];
        x_diffs = [x_diffs sum(differences(:,:,1))]; % sum of all dimensions, to give 2D dissim measure
        y_diffs = [y_diffs sum(differences(:,:,2))];
        
    end
    [x_diffs' y_diffs'];
end

function x_min  = radius_diss_shift(dissims, x_matrix, sigma_n)
    % Get gp for this specific radius
    [par, fval, flag] = fminunc(@(mypar)gp_max_log_like(mypar(1), mypar(2), sigma_n,...
                                                        dissims' , x_matrix(:,1)),...
                                [7 2] ,optimoptions('fminunc','Display','off'));
    
    sigma_f = par(1);
    l = par(2);


    k_cap = calc_k_cap(x_matrix(:,1), sigma_f,l, sigma_n);

    % Estimate position over length of radius
    i = 1;
    for x_star = -10:0.1:10
%         if sum(-20:10 == x_star) == 0
            x_stars(i) = x_star;

            % setup covariance matrix stuff
            k_star      = calc_k_star(x_star, x_matrix(:,1), sigma_f,l, sigma_n);
            k_star_star = calc_covar_ij(x_star, x_star, sigma_f,l);

            % Estimate y
            y_star(i) = k_star * inv(k_cap) * dissims';

            % Estimate variance
            var_y_star(i) = k_star_star - (k_star * inv(k_cap) * transpose(k_star));
            if var_y_star(i) < 0.0000
                var_y_star(i) =0; % otherwise -0.0000 causes errors with sqrt()
            end

            i = i+1;
%         end
    end
    figure(2)
%     clf
    hold on
%     fill([x_stars, fliplr(x_stars)],...
%          [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
%          [1 1 0.8])
    
    plot(x_matrix(:,1), dissims, '+')
    plot(x_stars, y_star)
    hold off
    
    [~,x_min_ind] = min(y_star);
    x_min = -x_stars(x_min_ind);
    
    figure(4)
    hold on
    plot(x_matrix(:,1)+x_min, dissims, '+')
    plot(x_stars+x_min, y_star)
    grid on
    grid minor
    axis([-20 10 0 75])
    hold off
%     y_for_real

%     
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

function test_plot(x_matrix, y, sigma_f, l_disp, l_mu, sigma_n,ref_tap, ref_diffs_norm,ref_diffs_norm_max_ind)
    l = [l_disp, l_mu];
    k_cap = calc_k_cap(x_matrix, sigma_f, l, sigma_n);

    i = 1;
    for mu_star = [0 1 2 3]
        for disp_star = -10:1:10
    %         if sum(-20:10 == x_star) == 0
                x_star = [disp_star mu_star];
                x_stars(i,:) = x_star;

                % setup covariance matrix stuff
                k_star      = calc_k_star(x_star, x_matrix, sigma_f,l, sigma_n);
                k_star_star = calc_covar_ij(x_star, x_star, sigma_f,l);

                % Estimate y
                y_star(i,:) = k_star * inv(k_cap) * y;

                % Estimate variance
                var_y_star(i) = k_star_star - (k_star * inv(k_cap) * transpose(k_star));
                if var_y_star(i) < 0.0000
                    var_y_star(i) =0; % otherwise -0.0000 causes errors with sqrt()
                end

                i = i+1;
    %         end
        end
    end
    figure(1)
    clf
%     hold on
%     fill([x_stars, fliplr(x_stars)],...
%          [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
%          [1 1 0.8])
    colour_graph = 'b'
    for num_plot = 1:size(y_star,1)
        clf
%         subplot(2,1,1)
        if num_plot == 21
            colour_graph = 'r'
        end
        if num_plot == 42
            colour_graph = 'b'
        end
        if num_plot == 63
            colour_graph = 'r'
        end
        
        xlabel(["Num" mod(num_plot,21)-11])
        
        hold on
%         plot(ref_tap(1 ,:  ,1),ref_tap(1 ,:  ,2),'+','Color','k')
%         plot(ref_tap(1 ,:  ,1)+ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,1),ref_tap(1 ,:  ,2)+ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,2),'+','Color','b')
        
%         plot(y_star(num_plot,1:127),y_star(num_plot,128:end),'o','Color','r')
        plot(y_star(num_plot,1:127)+ref_tap(1 ,:  ,1),y_star(num_plot,128:end)+ref_tap(1 ,:  ,2),'o','Color','r')
%         plot(y_star(num_plot,1:127)+ref_tap(1 ,:  ,1),y_star(num_plot,128:end)+ref_tap(1 ,:  ,2),'Color','r')

%         hold off
%         subplot(2,1,2)
%         
%         xlabel(["Num" num_plot-11])
%         hold on
%         plot(y(round(num_plot/2),1:127),y(round(num_plot/2),128:end),'o','Color','b')
%         plot(y(round(num_plot/2),1:127),y(round(num_plot/2),128:end),'o','Color',colour_graph)
%         axis([-10 10 -10 10])
%         axis([-10 10 -10 10])

%         plot(y(round(num_plot/2),1:127)+ref_tap(1 ,:  ,1),y(round(num_plot/2),128:end)+ref_tap(1 ,:  ,2),'o','Color',colour_graph)
        plot(y(round(num_plot/2),1:127)+ref_tap(1 ,:  ,1),y(round(num_plot/2),128:end)+ref_tap(1 ,:  ,2),'o','Color','b')
        if mod(num_plot,21)-11 == -10
            pause(1)
        end
        pause(0.5)
        hold off
        
    end
%     plot(x_matrix(:,1), y, '+')
%     plot(x_stars, y_star)
%     hold off
%     [~,x_min_ind] = min(y_star)
%     x_shift = -x_stars(x_min_ind)
%     
%     figure(4)
%     hold on
%     plot(x_matrix(:,1)+x_shift, dissims, '+')
%     plot(x_matrix(:,1)+x_shift, dissims)
%     grid on
%     grid minor
%     axis([-20 10 0 75])
%     hold off
%     y_for_real

end