%% Clear stuff
clearvars
figure(1)
clf
figure(2)
clf
figure(3)
clf
clear all

%% Load data from repeated single radii exp.
% x will be displacement (later also angle/depth)
% y will be normalized tap at max disp (raw data minus first frame, max
% disps of those, diff with reference tap).

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
max_num = 19;

TRIM_DATA = true;
TRAIN_MIN_DISP = -10;


% Note, for all, max = 21, min = 1
min_index_test = 1;
max_index_test = 21;
min_i_train = 1;
max_i_train = 21;

x_real(:,1) = [-10:1:10]';
x_real_test = [-10:1:10]';

%% Define reference tap & stuff
ref_tap = all_data{10}{11};%(:,:,:); 

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));

%% Filter raw data (normalize, find max frame)
sigma_n_y = 1.94;
sigma_n_diss = 5;%0.5;%1.94;

i_trainings = [10 15 19 5 1];
for num_training = 1:5
    
    [dissims{num_training},...
     y_train{num_training},...
     x_diffs{num_training},...
     y_diffs{num_training}] = process_taps(all_data{i_trainings(num_training)},...
                                           ref_diffs_norm,...
                                           ref_diffs_norm_max_ind,...
                                           sigma_n_diss,... 
                                           x_real(:,1),...
                                           ref_tap);
                                                        
    x_mins{num_training}  = radius_diss_shift(dissims{num_training}, x_real(:,1), sigma_n_diss,TRAIN_MIN_DISP);
    if num_training == 1
        if x_mins{1} ~= 0
            warning("Reference tap is not the min at disp 0")
            x_mins{1}
        end
    else
        x_real(:,num_training) = x_real(:,1) + x_mins{num_training} ; % so all minima are aligned
        if TRIM_DATA
            x_real(:,num_training) = (x_real(:,num_training) >TRAIN_MIN_DISP).* x_real(:,num_training) + (x_real(:,num_training)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
        end
    end
 
end


%% Optimize hyper-params for training

init_hyper_pars = [7 2 2];
size_x2 = size(x_real(min_i_train:max_i_train,1));

y_gplvm_input = [y_train{1}(min_i_train:max_i_train,:);...
                 y_train{2}(min_i_train:max_i_train,:);...
                 y_train{3}(min_i_train:max_i_train,:);...
                 y_train{4}(min_i_train:max_i_train,:);...
                 y_train{5}(min_i_train:max_i_train,:)];

x_gplvm_input = [ [x_real(min_i_train:max_i_train,1);...
                   x_real(min_i_train:max_i_train,2);...
                   x_real(min_i_train:max_i_train,3);...
                   x_real(min_i_train:max_i_train,4);...
                   x_real(min_i_train:max_i_train,5)] ...
                  [zeros(size_x2);...
                   1*ones(size_x2);...
                   2*ones(size_x2);...
                   -1*ones(size_x2);...
                   -2*ones(size_x2)] ];

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
    warning("A hyper-parameter is zero! Probably a bad fit")
end

sigma_f = par(1)
l_disp = par(2)
l_mu = par(3)

%% Visualize learnt model
test_plot(x_gplvm_input, y_gplvm_input, sigma_f, l_disp, l_mu, sigma_n_y,ref_tap,ref_diffs_norm,ref_diffs_norm_max_ind)

%% Test model learnt
fprintf('Calculating predictions\n')
n_bad_flags = 0;
n_flags = 0;

% for each radius
for set_num = 1:max_num
    
    [dissims_test,...
     y_test,...
     x_diffs_test,...
     y_diffs_test] = process_taps(all_data{set_num},...
                                  ref_diffs_norm,...
                                  ref_diffs_norm_max_ind,...
                                  sigma_n_diss,...
                                  x_real(:,1),...
                                  ref_tap);
    if set_num ~= 1
        x_min  = radius_diss_shift(dissims_test, x_real(:,1), sigma_n_diss,TRAIN_MIN_DISP);

        x_real(:,set_num) = x_real(:,1) + x_min;
    end
    if TRIM_DATA
        x_real(:,set_num) = (x_real(:,set_num) >=TRAIN_MIN_DISP).* x_real(:,set_num)...
                            + (x_real(:,set_num)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
    end
        
    
    init_latent_vars = [0];
%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fmincon(fun,x0,A,b,Aeq,beq,lb,ub)

    [par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f,...
                                                              [l_disp l_mu],...
                                                              sigma_n_y,...
                                                              [y_gplvm_input; y_test(min_index_test:max_index_test,:)],...
                                                              [x_gplvm_input; [min_index_test:max_index_test]' ones(size(min_index_test:max_index_test))'*opt_pars(1)]),...
                                init_latent_vars,...
                                optimoptions('fminunc','Display','off'));

% 
%     [par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f,...
%                                                               [l_disp l_mu],...
%                                                               sigma_n_y,...
%                                                               [y_gplvm_input; y_gplvm_input(1:21,:)],...
%                                                               [x_gplvm_input; x_gplvm_input(22:42,1) ones(21,1)*opt_pars(1) ]),...
%                                 init_latent_vars,...
%                                 optimoptions('fminunc','Display','off'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if flag ~= 1
        n_bad_flags = n_bad_flags +1;
    end 

    new_x(set_num, min_index_test:max_index_test,:) = par;
    dissims_tests(set_num, min_index_test:max_index_test) = dissims_test(min_index_test:max_index_test);

    n_flags = n_flags +1;

    fprintf('.');
    
    fprintf('\n')
end

% Print things out
new_x
dissims_tests

n_bad_flags
n_flags

%% Plot everything

figure(3)
subplot(1,2,1)
hold on
title("Train & Test offline 3D")

plot3(x_real(min_index_test:max_index_test,1:end),...
      new_x(:,min_index_test:max_index_test,1)',...
      dissims_tests(:,min_index_test:max_index_test)')

surf([x_gplvm_input(4*size(x_real,1)+1:5*size(x_real,1),1)...
      x_gplvm_input(3*size(x_real,1)+1:4*size(x_real,1),1)...
      x_gplvm_input(1:size(x_real,1),1)...
      x_gplvm_input(size(x_real,1)+1:2*size(x_real,1),1)...
      x_gplvm_input(2*size(x_real,1)+1:3*size(x_real,1),1)],...
     [x_gplvm_input(4*size(x_real,1)+1:5*size(x_real,1),2)...
      x_gplvm_input(3*size(x_real,1)+1:4*size(x_real,1),2)...
      x_gplvm_input(1:size(x_real,1),2)...
      x_gplvm_input(size(x_real,1)+1:2*size(x_real,1),2)...
      x_gplvm_input(2*size(x_real,1)+1:3*size(x_real,1),2)],...
     [dissims{5}(min_i_train:max_i_train)'...
      dissims{4}(min_i_train:max_i_train)'...
      dissims{1}(min_i_train:max_i_train)'...
      dissims{2}(min_i_train:max_i_train)'...
      dissims{3}(min_i_train:max_i_train)'])

xlabel("Real Displacemt / mm")
zlabel("Dissim")
ylabel("mu")
hold off
grid on
grid minor


subplot(1,2,2)
hold on
title("Train & Test offline 2D")

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
        current_tap_data_norm = radii_data{1,tap_num}(: ,:  ,:)- radii_data{1,tap_num}(1 ,:  ,:);
%         current_tap_data_norm = radii_data{1,tap_num}(: ,:  ,:) - ref_tap(1 ,:  ,:);

%         diff_between_noncontacts = ref_tap(1 ,:  ,:) - radii_data{1,tap_num}(1 ,:  ,:); %TODO throw error if too large?
        
        n_pins = size(radii_data{1,1},2);
        max_i = zeros(1, n_pins);
        for pin = 1:n_pins

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
            y_for_real = [y_for_real;...
                          radii_data{1,tap_num}(average_max_i,:,1)  radii_data{1,tap_num}(average_max_i,:,2)];
        elseif Y_NORM
            y_for_real = [y_for_real;...
                          current_tap_data_norm(average_max_i,:,1) current_tap_data_norm(average_max_i,:,2)];
        elseif Y_NORM_DIFF
            y_for_real = [y_for_real;...
                          differences(:,:,1) differences(:,:,2)];
        else
            error("Y not chosen")
        end
        diss = norm([differences(:,:,1)'; differences(:,:,2)']);
%         diss = pdist2(differences(:,:,1),differences(:,:,2), 'euclidean');
        
        dissims =[dissims diss];
%         x_diffs = [x_diffs sum(differences(:,:,1))]; % sum of all dimensions, to give 2D dissim measure
%         y_diffs = [y_diffs sum(differences(:,:,2))];
        
    end
    [x_diffs' y_diffs'];
end

function x_min  = radius_diss_shift(dissims, x_matrix, sigma_n, TRAIN_MIN_DISP)
% Return number that when added to the suggested x values, shifts the
% values so that the trough (minima) lines up with 0. Uses a gp to estimate
% smooth curve rather than using raw dissim values (gp may need tuning
% under different circs e.g. harder taps giving higher dissims).Plot raw
% and gp estimates for reference.

    % Get gp for this specific radius
    [par, fval, flag] = fminunc(@(mypar)gp_max_log_like(mypar(1), mypar(2), sigma_n,...
                                                        dissims' , x_matrix(:,1)),...
                                [10 1] ,optimoptions('fminunc','Display','off'));
    
    sigma_f = par(1);
    l = par(2);

    % Get K matrix for this radius
    k_cap = calc_k_cap(x_matrix(:,1), sigma_f,l, sigma_n);

    % Estimate position over length of radius
    i = 1;
    for x_star = TRAIN_MIN_DISP:0.1:10
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
    
    [~,x_min_ind] = min(y_star);
    x_min = -x_stars(x_min_ind);
    
    %% Plot input
    figure(2)
    subplot(1,2,1)
%     clf
    hold on
    title("Original")
%     fill([x_stars, fliplr(x_stars)],...
%          [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
%          [1 1 0.8])
    
    plot(x_matrix(:,1), dissims, '+')
    plot(x_matrix(:,1), dissims)
    plot(x_stars, y_star)
    axis([-10 10 0 90])
    hold off
    
    %% Plot output
    subplot(1,2,2)
    hold on
    title("Troughs aligned")
    plot(x_matrix(:,1)+x_min, dissims, '+')
    plot(x_stars+x_min, y_star)
    grid on
    grid minor
    axis([-10 10 0 90])
    hold off

end

function test_plot(x_matrix, y, sigma_f, l_disp, l_mu, sigma_n,ref_tap, ref_diffs_norm,ref_diffs_norm_max_ind)
    l = [l_disp, l_mu];
    k_cap = calc_k_cap(x_matrix, sigma_f, l, sigma_n);

    i = 1;
    for mu_star = [0 1 2 -1 -2]
        for disp_star = -5:1:10
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
    var_y_star
    
    figure(1)
    clf
%     hold on
%     fill([x_stars, fliplr(x_stars)],...
%          [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
%          [1 1 0.8])
    colour_graph = 'b'
%     plot(y_star(:,1:126),y_star(:,127:end))
hold on
pc=plot(ref_tap(1 ,:  ,1),ref_tap(1 ,:  ,2),'+','Color','k');
% pa=plot(y_star(:,1:126)+ref_tap(1 ,:  ,1),y_star(:,127:end)+ref_tap(1 ,:  ,2),'+','Color','r'); % predicted
pa=plot(y_star(:,1:126)+ref_tap(1 ,:  ,1),y_star(:,127:end)+ref_tap(1 ,:  ,2),'Color','r'); % predicted
% pb=plot(y(:,1:126)+ref_tap(1 ,:  ,1),y(:,127:end)+ref_tap(1 ,:  ,2),'+','Color','b'); % actual
pb=plot(y(:,1:126)+ref_tap(1 ,:  ,1),y(:,127:end)+ref_tap(1 ,:  ,2),'Color','b'); % actual
axis equal
title("Real vs. predicted taxel locations")
xlabel("x displacement / pixels")
ylabel("y displacement / pixels")
legend([pc pb(1) pa(1)],{'Non-contact Location','Real','Predicted'})
text(370,-10,'Data set: singleRadius2019-01-16\_1651')
hold off

% plot(y_star(:,1:126),y_star(:,127:end),'Color','r') % predicted
% plot(y(:,1:126),y(:,127:end),'Color','b') % actual

    if false
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
            title("Taxel patterns")

            % Reference frame
    %         plot(ref_tap(1 ,:  ,1),ref_tap(1 ,:  ,2),'+','Color','k')
    %         plot(ref_tap(1 ,:  ,1)+ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,1),ref_tap(1 ,:  ,2)+ref_diffs_norm(ref_diffs_norm_max_ind ,:  ,2),'+','Color','b')

            % Predicted tap
    %         plot(y_star(num_plot,1:126),y_star(num_plot,127:end),'o','Color','r')
    %         plot(y_star(num_plot,1:126)+ref_tap(1 ,:  ,1),y_star(num_plot,127:end)+ref_tap(1 ,:  ,2),'o','Color','r')

            % Actual tap 
    %         plot(y(num_plot,1:126)+ref_tap(1 ,:  ,1),y(num_plot,127:end)+ref_tap(1 ,:  ,2),'o','Color','b')
            plot(y(num_plot,1:126),y(num_plot,127:end),'o','Color','b')
    %         axis([0 550 0 550])
            axis([-30 30 -60 60])
            if mod(num_plot,21)-11 == -10
                pause(1)
            end
            pause(0.5)

            hold off

        end
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