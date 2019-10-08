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
    for i_angle = 1:n_angles
        for disp = 1:n_disps_per_radii
            all_data{depth}{i_angle}{1,disp}= data{1,current_number};
            current_number = current_number + 1;
        end
    end
end
all_data{1};


TRIM_DATA = false;
TRAIN_MIN_DISP = -10;


% Note, for all, max = 21, min = 1
TEST_RANGE = 1:21;
TRAIN_RANGE = 1:21;

x_real_train(:,1) = [-10:1:10]';
x_real_test(:,1) = [-10:1:10]';
X_SHIFT_ON = false;
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
training_depth_indexes = [3 5 7];%4:6;%1:9;
training_depths = (training_depth_indexes-5)/2 ; 

i_trainings = round(linspace(1,19,n_training_angles));%[10 15 19 5 1];
i_train_data = 0;
i_depths = 0;
for training_depth = training_depth_indexes
    i_depths = i_depths+1;
    for angle_num = 1:n_training_angles
        i_train_data = i_train_data +1;

        [dissims{i_depths}{angle_num},...
         y_train{i_depths}{angle_num},...
         x_diffs{i_depths},...
         y_diffs{i_depths}] = process_taps(all_data{training_depth}{i_trainings(angle_num)},...
                                               ref_diffs_norm,...
                                               ref_diffs_norm_max_ind,...
                                               sigma_n_diss,... 
                                               x_real_train(:,1),...
                                               ref_tap);
        
        
        if X_SHIFT_ON
            x_mins{i_train_data}  = radius_diss_shift(dissims{i_depths}{angle_num}, x_real_train(:,1), sigma_n_diss,TRAIN_MIN_DISP,training_depth,angle_num==1);
            if angle_num == 1
                if x_mins{1} ~= 0
                    warning("Reference tap is not the min at disp 0")
                    x_mins{1}
                end
            else
                if TRIM_DATA
                        x_real_train(:,i_train_data) = (x_real_train(:,i_train_data) >TRAIN_MIN_DISP).* x_real_train(:,i_train_data) + (x_real_train(:,i_train_data)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
                end
            end
            x_real_train(:,i_train_data) = x_real_train(:,1) + x_mins{i_train_data} ; % so all minima are aligned
        else
            x_real_train(:,i_train_data) = x_real_train(:,1);               
        end
            
        
        
    
    end
end


%% Optimize hyper-params for training

init_hyper_pars = [1 300 5];
size_x2 = size(x_real_train(TRAIN_RANGE,1));


% y_gplvm_input_train = [y_train{1}(TRAIN_RANGE,:);...
%                  y_train{2}(TRAIN_RANGE,:);...
%                  y_train{3}(TRAIN_RANGE,:);...
%                  y_train{4}(TRAIN_RANGE,:);...
%                  y_train{5}(TRAIN_RANGE,:)
% ];

y_gplvm_input_train=[];
disp_gplvm_input_train=[];
% Build matrix of y and x for use in gplvm

for depth = 1:i_depths
    for angle_num = 1:n_training_angles
    y_gplvm_input_train = [y_gplvm_input_train;...
                           y_train{depth}{angle_num}(TRAIN_RANGE,:)];
    end
end

for indexes = 1:i_train_data                   
    disp_gplvm_input_train = [disp_gplvm_input_train;...
                           x_real_train(TRAIN_RANGE,indexes)];                   
end


% [x_real(TRAIN_RANGE,1);...
%                                                                             x_real(TRAIN_RANGE,2);...
%                                                                             x_real(TRAIN_RANGE,3);...
%                                                                             x_real(TRAIN_RANGE,4);...
%                                                                             x_real(TRAIN_RANGE,5)
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
for depths = 1:i_depths
    for indexes = linspace(-2,2,n_training_angles)
    mu_gplvm_input_train = [mu_gplvm_input_train;...
                           indexes*ones(size_x2)
    ];
    end
end

depth_gplvm_input_train = reshape(repmat(training_depths,n_disps_per_radii*n_training_angles,1), length(mu_gplvm_input_train),1);

x_gplvm_input_train = [disp_gplvm_input_train mu_gplvm_input_train depth_gplvm_input_train];
%% %%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
fprintf('Training GPLVM\n')
init_hyper_pars_2 = [1 300 5 0.5];

[par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(opt_pars(1), ...
                                                          [opt_pars(2) opt_pars(3) opt_pars(4)], ...
                                                          sigma_n_y,...
                                                          y_gplvm_input_train ,...
                                                          x_gplvm_input_train),...
                            init_hyper_pars_2,...
                            optimoptions('fminunc','Display','off','MaxFunEvals',10000));
fprintf('Finished training GPLVM\n')                        
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
l_depth = par(4)
% par(4)
% par(5)



%% Visualize learnt model
%test_plot(x_gplvm_input_train, y_gplvm_input_train, sigma_f, l_disp, l_mu, sigma_n_y,ref_tap,ref_diffs_norm,ref_diffs_norm_max_ind)

%% Test model learnt
fprintf('Calculating predictions\n')
n_bad_flags = 0;
n_flags = 0;

test_depth_indexes = 1:9;
test_depths = (test_depth_indexes-5)/2;
i_tests =0;

%for each depth 
for test_depth = test_depth_indexes
    % for each radius
    for i_angle = 1:n_angles
        i_tests = i_tests +1;
        [dissims_test,...
         y_test,...
         x_diffs_test,...
         y_diffs_test] = process_taps(all_data{test_depth}{i_angle},...
                                      ref_diffs_norm,...
                                      ref_diffs_norm_max_ind,...
                                      sigma_n_diss,...
                                      x_real_test(:,1),...
                                      ref_tap);
        
        if X_SHIFT_ON
            if i_tests ~= 1
                x_min  = radius_diss_shift(dissims_test, x_real_test(:,1), sigma_n_diss,TRAIN_MIN_DISP,test_depth,true);
                x_real_test(:,i_tests) = x_real_test(:,1) + x_min;
            end
        else
            x_real_test(:,i_tests) = x_real_test(:,1);
        end
        
        if TRIM_DATA
            x_real_test(:,i_tests) = (x_real_test(:,i_tests) >=TRAIN_MIN_DISP).* x_real_test(:,i_tests)...
                                + (x_real_test(:,i_tests)<TRAIN_MIN_DISP).* TRAIN_MIN_DISP;
        end


        init_latent_vars = [0 0];
    %%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % fprintf('Testing GPLVM\n')
    % fmincon(fun,x0,A,b,Aeq,beq,lb,ub)

        [par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f,...
                                                                  [l_disp l_mu l_depth],...
                                                                  sigma_n_y,...
                                                                  [y_gplvm_input_train; y_test(TEST_RANGE,:)],...
                                                                  [x_gplvm_input_train; x_real_test(TEST_RANGE,i_tests) ones(size(TEST_RANGE))'*opt_pars(1) ones(size(TEST_RANGE))'*opt_pars(2)]),...
                                    init_latent_vars,...
                                    optimoptions('fminunc','Display','off','MaxFunEvals',10000));

    % 
    %     [par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f,...
    %                                                               [l_disp l_mu],...
    %                                                               sigma_n_y,...
    %                                                               [y_gplvm_input; y_gplvm_input(1:21,:)],...
    %                                                               [x_gplvm_input; x_gplvm_input(22:42,1) ones(21,1)*opt_pars(1) ]),...
    %                                 init_latent_vars,...
    %                                 optimoptions('fminunc','Display','off'));

    % fprintf('Finished testing GPLVM\n')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if flag < 1
            n_bad_flags = n_bad_flags +1;
            flag
        end 

        new_mu(i_tests, TEST_RANGE,:) = par(1);
        new_depth(i_tests, TEST_RANGE,:) = par(2);
        dissims_tests(i_tests, TEST_RANGE) = dissims_test(TEST_RANGE);

        n_flags = n_flags +1;

        fprintf('.');
    end
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
subplot(1,3,2)
hold all
title_string = strcat("Train & Test Offline \mu Error: No. training angles=", num2str(n_training_angles), ", No.training depths=", num2str(length(training_depth_indexes)));
title(title_string)
plot([-2 2],[0 0],'k')
% plot(x_real(TEST_RANGE,1:end),new_mu(:,TEST_RANGE,1)','+')
% plot(x_real(TEST_RANGE,1:end),new_mu(:,TEST_RANGE,1)')
% plot(x_gplvm_input_train(:,1),x_gplvm_input_train(:,2),'o')
% xlabel("Real Displacemt / mm")
% ylabel("mu / mm")

expected_mu_block = repmat([-2:4/18:2],1,length(test_depth_indexes));%-2:4/18:2;
expected_mu = -2:4/18:2;
for test_depth_i = 1:length(test_depth_indexes)
    %hold on
%     hold on
    stem(expected_mu,new_mu(((test_depth_i-1)*n_angles)+1:(test_depth_i)*n_angles,1,1)'-expected_mu,'x')
end
mu_mean_error = mean(abs(new_mu(:,1,1)'-expected_mu_block))
plot(x_gplvm_input_train(:,2),x_gplvm_input_train(:,2)-x_gplvm_input_train(:,2),'ok','MarkerFaceColor','r') %show where training lines are
%axis([-2.2 2.2 -0.171 0.33 ])
axis([-2.2 2.2 -3 3 ])

ylabel("Error in predicted \mu")
xlabel("Expected \mu")
grid on
grid minor
hold off


subplot(1,3,3)
hold all
title_string = strcat("Train & Test Offline depth Error: No. training angles=", num2str(n_training_angles), ", No.training depths=", num2str(length(training_depth_indexes)));
title(title_string)
plot([-2 2],[0 0],'k')

test_depths
new_depth
% expected_depth_block = repmat([-2:4/18:2],1,length(test_depths));%-2:4/18:2;
expected_depths = reshape(repmat(test_depths,n_angles,1), length(test_depths)*n_angles,1);

differences = new_depth(:,1) - expected_depths;

stem(expected_depths,differences,'x')

depth_mean_error = mean(abs(differences))
% plot(x_gplvm_input_train(:,2),x_gplvm_input_train(:,2)-x_gplvm_input_train(:,2),'ok','MarkerFaceColor','r') %show where training lines are
%axis([-2.2 2.2 -0.171 0.33 ])
% axis([-2.2 2.2 -3 3 ])

ylabel("Error in predicted depth")
xlabel("Expected depth")
grid on
grid minor
hold off


subplot(1,3,1)
hold on
title_string = strcat("Train & Test Offline 3D: No. training angles=", num2str(n_training_angles), ", No.training depths=", num2str(length(training_depth_indexes)));
title(title_string)

% plot3(x_real(TEST_RANGE,1:end),...
%       new_mu(:,TEST_RANGE,1)',...
%       dissims_tests(:,TEST_RANGE)')
  
surf(x_real_test(TEST_RANGE,1:end),...
      new_mu(:,TEST_RANGE,1)',...
      dissims_tests(:,TEST_RANGE)')
  
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
%      [dissims{5}(TRAIN_RANGE)'...
%       dissims{4}(TRAIN_RANGE)'...
%       dissims{1}(TRAIN_RANGE)'...
%       dissims{2}(TRAIN_RANGE)'...
%       dissims{3}(TRAIN_RANGE)'])

xlabel("Displacemt (mm)")
zlabel("Dissim")
ylabel("\mu")
hold off
grid on
grid minor




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

function x_min  = radius_diss_shift(dissims, x_matrix, sigma_n, TRAIN_MIN_DISP, depth,legend_on)
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
    colour = [(mod(depth,2)) 1-(mod(depth,9)/9) mod(depth,9)/9];
    plot(x_matrix(:,1), dissims, '+','Color',colour,'HandleVisibility','off')
    plot(x_matrix(:,1), dissims,'Color',colour,'HandleVisibility','off')
    actual_depth = 2.5-(depth/2);
    if legend_on
        plot(x_stars, y_star,'Color',colour,'DisplayName',['depth=' num2str(actual_depth)]);
    else
        plot(x_stars, y_star,'Color',colour,'HandleVisibility','off');
    end
%     name_d = strcat('depth=',num2str(depth))
    legend('show')%([p(depth)],{name_d})
%     legend(gca,'off');    
%     legend('show');
    axis([-10 10 0 90])
    grid on
    grid minor
    hold off
    
    %% Plot output
    subplot(1,2,2)
    hold on
    title("Troughs aligned")
    plot(x_matrix(:,1)+x_min, dissims, '+','Color',colour)
    plot(x_stars+x_min, y_star,'Color',colour)
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