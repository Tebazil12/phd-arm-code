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



% load('/home/lizzie/git/masters-tactile/data/partCircleRadii2019-01-09_1421/c45_01.mat') % whole circ, fixed angle wrt workframe
load('/home/lizzie/OneDrive/data/collect_data_3d_varyAngle_FIXEDslice2019-10-01_1901/c45_01_20.mat')
%all_data = data; % get to state all_data{depth}{angle}{disp}

n_disps_per_radii = 21;
n_angles = 19;
n_depths = 9;
current_number = 1;
for depth = 1:n_depths
    for angle = 1:n_angles
        for disp = 1:n_disps_per_radii
            all_data{depth}{angle}{1,disp}= data{1,current_number};
            current_number = current_number + 1;
        end
    end
end
all_data{1};

% n_angles = 19;


TRIM_DATA = true;
TRAIN_MIN_DISP = -10;


% Note, for all, max = 21, min = 1
MIN_I_TEST = 1;
MAX_I_TEST = 21;
MIN_I_TRAIN = 1;
MAX_I_TRAIN = 21;

x_real(:,1) = [-10:1:10]';
x_real_test = [-10:1:10]';
dissims =[];

%% Define reference tap & stuff
ref_tap = all_data{5}{10}{11};%(:,:,:); 

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));

%% Filter raw data (normalize, find max frame)
sigma_n_y = 1.14;%1.94;
sigma_n_diss = 5;%0.5;%1.94;

n_training_angles = 5;

i_trainings = round(linspace(1,19,n_training_angles));%[10 15 19 5 1];
i_train_data = 1;
for training_depths = 4:6%4:6
    for num_training = 1:n_training_angles

        [dissims{i_train_data},...
         y_train{i_train_data},...
         x_diffs{i_train_data},...
         y_diffs{i_train_data}] = process_taps(all_data{training_depths}{i_trainings(num_training)},...
                                               ref_diffs_norm,...
                                               ref_diffs_norm_max_ind,...
                                               sigma_n_diss,... 
                                               x_real(:,1),...
                                               ref_tap);

        x_mins{i_train_data}  = radius_diss_shift(dissims{i_train_data}, x_real(:,1), sigma_n_diss,TRAIN_MIN_DISP);
        if num_training == 1
            if x_mins{1} ~= 0
                warning("Reference tap is not the min at disp 0")
                x_mins{1}
            end
        else
            x_real(:,i_train_data) = x_real(:,1) + x_mins{i_train_data} ; % so all minima are aligned
            if TRIM_DATA
                x_real(:,i_train_data) = (x_real(:,i_train_data) >TRAIN_MIN_DISP).* x_real(:,i_train_data) + (x_real(:,i_train_data)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
            end
        end
    i_train_data = i_train_data +1;
    end
    
end


%% Optimize hyper-params for training

init_hyper_pars = [1 300 5];
size_x2 = size(x_real(MIN_I_TRAIN:MAX_I_TRAIN,1));


% y_gplvm_input_train = [y_train{1}(MIN_I_TRAIN:MAX_I_TRAIN,:);...
%                  y_train{2}(MIN_I_TRAIN:MAX_I_TRAIN,:);...
%                  y_train{3}(MIN_I_TRAIN:MAX_I_TRAIN,:);...
%                  y_train{4}(MIN_I_TRAIN:MAX_I_TRAIN,:);...
%                  y_train{5}(MIN_I_TRAIN:MAX_I_TRAIN,:)
% ];

y_gplvm_input_train=[];
disp_gplvm_input_train=[];
for indexes = 1:n_training_angles
y_gplvm_input_train = [y_gplvm_input_train;...
                       y_train{indexes}(MIN_I_TRAIN:MAX_I_TRAIN,:)];
disp_gplvm_input_train = [disp_gplvm_input_train;...
                       x_real(MIN_I_TRAIN:MAX_I_TRAIN,indexes)];                   
end


% [x_real(MIN_I_TRAIN:MAX_I_TRAIN,1);...
%                                                                             x_real(MIN_I_TRAIN:MAX_I_TRAIN,2);...
%                                                                             x_real(MIN_I_TRAIN:MAX_I_TRAIN,3);...
%                                                                             x_real(MIN_I_TRAIN:MAX_I_TRAIN,4);...
%                                                                             x_real(MIN_I_TRAIN:MAX_I_TRAIN,5)
% ] ...

for indexes = 1:n_training_angles

end

% [zeros(size_x2);...
%                                                                             1*ones(size_x2);...
%                                                                             2*ones(size_x2);...
%                                                                             -1*ones(size_x2);...
%                                                                             -2*ones(size_x2)
% ]
mu_gplvm_input_train=[];
for indexes = linspace(-2,2,n_training_angles)
mu_gplvm_input_train = [mu_gplvm_input_train;...
                       indexes*ones(size_x2)
];
end
%% %%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
init_hyper_pars_2 = [1 300 5];

[par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(opt_pars(1), ...
                                                          [opt_pars(2) opt_pars(3)], ...
                                                          sigma_n_y,...
                                                          y_gplvm_input_train ,[disp_gplvm_input_train ...
                                                                                mu_gplvm_input_train]),...
                            init_hyper_pars_2,...
                            optimoptions('fminunc','Display','off','MaxFunEvals',10000));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
flag
if flag < 1
    warning("fminsearch was not happy")
    flag
end 

if round(par(1),1) == 0 || round(par(2),1) == 0
    warning("A hyper-parameter is zero! Probably a bad fit")
end

par

sigma_f = par(1)
l_disp = par(2)
l_mu = par(3)
% par(4)
% par(5)
x_gplvm_input_train = [disp_gplvm_input_train mu_gplvm_input_train];


%% Visualize learnt model
test_plot(x_gplvm_input_train, y_gplvm_input_train, sigma_f, l_disp, l_mu, sigma_n_y,ref_tap,ref_diffs_norm,ref_diffs_norm_max_ind)

%% Test model learnt
fprintf('Calculating predictions\n')
n_bad_flags = 0;
n_flags = 0;

% for each radius
for set_num = 1:n_angles
    
    [dissims_test,...
     y_test,...
     x_diffs_test,...
     y_diffs_test] = process_taps(all_data{5}{set_num},...
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
                                                              [y_gplvm_input_train; y_test(MIN_I_TEST:MAX_I_TEST,:)],...
                                                              [x_gplvm_input_train; x_real(MIN_I_TEST:MAX_I_TEST,set_num) ones(size(MIN_I_TEST:MAX_I_TEST))'*opt_pars(1)]),...
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
    if flag < 1
        n_bad_flags = n_bad_flags +1;
        flag
    end 

    new_mu(set_num, MIN_I_TEST:MAX_I_TEST,:) = par;
    dissims_tests(set_num, MIN_I_TEST:MAX_I_TEST) = dissims_test(MIN_I_TEST:MAX_I_TEST);

    n_flags = n_flags +1;

    fprintf('.');
    
    fprintf('\n')
end

% Print things out
new_mu
dissims_tests

n_bad_flags
n_flags

%% Plot everything

figure(3)
clf
subplot(1,2,1)
hold on
title("Train & Test Offline 3D - 5 Input Lines")

% plot3(x_real(MIN_I_TEST:MAX_I_TEST,1:end),...
%       new_mu(:,MIN_I_TEST:MAX_I_TEST,1)',...
%       dissims_tests(:,MIN_I_TEST:MAX_I_TEST)')
  
surf(x_real(MIN_I_TEST:MAX_I_TEST,1:end),...
      new_mu(:,MIN_I_TEST:MAX_I_TEST,1)',...
      dissims_tests(:,MIN_I_TEST:MAX_I_TEST)')
  
view([-1,0.5,0.2])
% for i = 1:5
%   
%     plot3(x_gplvm_input_train(i,1), x_gplvm_input_train(i,2), dissims(i))
% end

% surf([x_gplvm_input_train(4*size(x_real,1)+1:5*size(x_real,1),1)...
%       x_gplvm_input_train(3*size(x_real,1)+1:4*size(x_real,1),1)...
%       x_gplvm_input_train(1:size(x_real,1),1)...
%       x_gplvm_input_train(size(x_real,1)+1:2*size(x_real,1),1)...
%       x_gplvm_input_train(2*size(x_real,1)+1:3*size(x_real,1),1)],...
%      [x_gplvm_input_train(4*size(x_real,1)+1:5*size(x_real,1),2)...
%       x_gplvm_input_train(3*size(x_real,1)+1:4*size(x_real,1),2)...
%       x_gplvm_input_train(1:size(x_real,1),2)...
%       x_gplvm_input_train(size(x_real,1)+1:2*size(x_real,1),2)...
%       x_gplvm_input_train(2*size(x_real,1)+1:3*size(x_real,1),2)],...
%      [dissims{5}(MIN_I_TRAIN:MAX_I_TRAIN)'...
%       dissims{4}(MIN_I_TRAIN:MAX_I_TRAIN)'...
%       dissims{1}(MIN_I_TRAIN:MAX_I_TRAIN)'...
%       dissims{2}(MIN_I_TRAIN:MAX_I_TRAIN)'...
%       dissims{3}(MIN_I_TRAIN:MAX_I_TRAIN)'])

xlabel("Displacemt (mm)")
zlabel("Dissim")
ylabel("\mu")
hold off
grid on
grid minor


subplot(1,2,2)
hold on
title("Train & Test Offline \mu Error - 5 Input Lines")

% plot(x_real(MIN_I_TEST:MAX_I_TEST,1:end),new_mu(:,MIN_I_TEST:MAX_I_TEST,1)','+')
% plot(x_real(MIN_I_TEST:MAX_I_TEST,1:end),new_mu(:,MIN_I_TEST:MAX_I_TEST,1)')
% plot(x_gplvm_input_train(:,1),x_gplvm_input_train(:,2),'o')
% xlabel("Real Displacemt / mm")
% ylabel("mu / mm")

expected_mu = -2:4/18:2;
bar(expected_mu,new_mu(:,1,1)'-expected_mu)
mean(abs(new_mu(:,1,1)'-expected_mu))
plot(x_gplvm_input_train(:,2),x_gplvm_input_train(:,2)-x_gplvm_input_train(:,2),'ok','MarkerFaceColor','r')
%axis([-2.2 2.2 -0.171 0.33 ])
axis([-2.2 2.2 -3 3 ])

ylabel("Error in predicted \mu")
xlabel("Expected \mu")
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
pa=plot(y_star(:,1:125)+ref_tap(1 ,:  ,1),y_star(:,126:end)+ref_tap(1 ,:  ,2),'Color','r'); % predicted
% pb=plot(y(:,1:126)+ref_tap(1 ,:  ,1),y(:,127:end)+ref_tap(1 ,:  ,2),'+','Color','b'); % actual
pb=plot(y(:,1:125)+ref_tap(1 ,:  ,1),y(:,126:end)+ref_tap(1 ,:  ,2),'Color','b'); % actual
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
            plot(y(num_plot,1:125),y(num_plot,126:end),'o','Color','b')
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