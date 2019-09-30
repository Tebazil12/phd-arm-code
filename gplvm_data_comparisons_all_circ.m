%% Load data from repeated single radii exp.
% x will be displacement (later also angle/depth)
% y will be normalized tap at max disp (raw data minus first frame, max
% disps of those, diff with reference tap).

clearvars
clf

load('/home/lizzie/git/masters-tactile/data/wholeCircleRadii2018-10-22_1615/c180_01.mat')

% load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_01.mat')
for radii_num = 1:10
    for num = 1:31
        actual_index = (31*(radii_num-1))+num;
        all_data{radii_num}{1,num}= data{1,actual_index};
    end
end
all_data{1}


% x_real = [3:-1:-3]';
% x_real = [10:-2:-4]';
x_real = [-20:10]';
x_real_test = [-20:10]';

%% Define reference tap & stuff
ref_tap = all_data{1}{1,21};%(:,:,:); 
% ref_tap = all_data{1}{1,6};%(:,:,:); 
% ref_tap = all_data{1}{1,4};%(:,:,:); 

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));

%% Filter raw data (normalize, find max frame)

[dissims, y_train] = process_taps(all_data{1},ref_diffs_norm,ref_diffs_norm_max_ind)


%% Optimize hyper-params for "training"

init_hyper_pars = [7 0.2 1.1467];
% init_latent_vars = [x_real];

[par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(opt_pars(1), opt_pars(2), opt_pars(3),...
                                                          y_train , x_real),...
                            init_hyper_pars)

if flag ~= 1
    warning("fminsearch was not happy")
    flag
end 

if round(par(1),1) == 0 || round(par(2),1) == 0 || round(par(3),1) == 0
    warning("A hyper-parameter is zero! Bad fit")
end

sigma_f = par(1);
l = par(2);
sigma_n = 1.1467;%par(3); %TODO this is wrong, implement higher up
% predicted_x = [par(4:end)]';
% predicted_x = [predicted_x'];

%% "Test" stuff
for test_tap_num = 1:31

    for set_num = 2:10
        [dissims_test, y_test] = process_taps(all_data{set_num},ref_diffs_norm,ref_diffs_norm_max_ind);

        % init_hyper_pars = [0.5 0.2 0.2];
%         init_latent_vars = [((10+1)-test_tap_num)+0.01]; %TODO obviously test what happens when these are shifted (whole mm or otherwise)
        init_latent_vars = [0];

        [par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f, l, sigma_n,...
                                                                  [y_train; y_test(test_tap_num,:)] , [x_real; opt_pars]),...
                                    init_latent_vars , optimoptions('fminunc','Display','off'));

        if flag ~= 1
            warning("fminsearch was not happy")
            flag
        end 

        new_x(set_num-1, test_tap_num) = par;
        dissims_tests(set_num-1, test_tap_num) = dissims_test(test_tap_num);
        x_error(set_num-1, test_tap_num) = par - (test_tap_num - 20);
    end
end

x_error
max_x_error = max(abs(x_error))


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
plot(x_real, dissims, 'k')
% scatter(x_real, dissims, 'k','+')

plot(x_real, dissims_tests,'r')
% scatter(x_real, dissims_test,'r','+')
% 
% plot(10:-1:-20, dissims_test,'r')
% scatter(10:-1:-20, dissims_test,'r','+')

% plot predicted data
plot(new_x,dissims_tests, '+') 
% scatter(new_x,dissims_tests, 'b') 

plot([(10+1)-test_tap_num (10+1)-test_tap_num],[0 30],'b')



grid on
grid minor

hold off
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
hold on
plot(x_real_test,new_x,'+')
plot([-20 10],[-20 10],'k')
xlabel("Real Displacemt / mm")
ylabel("Predicted Displacemt / mm")
hold off

grid on
grid minor

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------- FUNCTIONS ---------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function val = gp_max_log_like(sigma_f, l, sigma_n, y, x_matrix)
%     sigma_f
%     l
%     sigma_n
    k_cap = calc_k_cap(x_matrix, sigma_f, l, sigma_n);
    if det(k_cap) == 0 
        error("Determinant was 0, unable to continue")
    end
    
    part1 = -0.5 * y' * pinv(k_cap) * y;
    part2 = -0.5 * log(det(k_cap)); % NB, |K| means det(K), not abs(K) or norm(K)
    part3 = -0.5 * length(y)*log(2*pi);
    neg_val = part1 + part2 + part3;
    val = - neg_val; % because trying to find max with a min search

%     [n, ~] = size(k_cap);
%     R = chol(k_cap); % K = R'*R
%     y';
%     % Calculate the log determinant of the covariance matrix
%     logdet_K = sum(2*log(R(1:n+1:end)));
% 
% 
%     % Calculate the log likelihood
%     alpha = R \ (R' \ y);
% %     val = 0.5*mean(sum(y'.*alpha, 1)) + 0.5*logdet_K + 0.5*n*log(2*pi);
%     val = (0.5* y' * (R' * R)\y )+ 0.5*logdet_K + 0.5*n*log(2*pi);

end

function val = gplvm_max_log_like(sigma_f, l, sigma_n, y, x_matrix)
%     sigma_f
%     l
%     sigma_n
    x_matrix;
    
    if sum(x_matrix(end) == x_matrix) > 1
%         warning("repetition of x exactly is not allowed, adding 0.0000000000001")
        x_matrix(end) = x_matrix(end) +0.0000000000001;
    end

    [d_cap, n_cap] = size(y);
    s_cap = get_s_cap(d_cap, y);

    k_cap = calc_k_cap(x_matrix, sigma_f, l, sigma_n);
    
%     R = chol(k_cap)
    
    if det(k_cap) == 0 
        error("Determinant was 0, unable to continue")
    end
    
    part1 = - (d_cap * n_cap) * 0.5 * log(2 * pi);
    part2 = - (d_cap) * 0.5 * log(det(k_cap)); % NB, |K| means det(K), not abs(K) or norm(K)
    part3 = - (d_cap) * 0.5 * trace(pinv(k_cap) * s_cap);
    
    neg_val = part1 + part2 + part3;
    val = - neg_val; % because trying to find max with a min search
end
 
function s_cap = get_s_cap(d_cap, y)
    s_cap = (1/d_cap) * (y * y');
%     s_cap = zeros(n_cap,n_cap);
%     for i = 1:n_cap 
%         for j = 1:n_cap
%             s_cap(i,j) = calc_covar_ij(x_matrix(i),x_matrix(j), sigma_f,l, sigma_n);
%             
%         end
%     end
%     s_cap
    
end

% Define covariance function, k:
function k = calc_covar_ij(x,x_prime,sigma_f,l, sigma_n)
    if x == x_prime
        kron_delta_func = 1;
    else
        kron_delta_func = 0;
    end
    k = (sigma_f^2 * exp( (-(x - x_prime)^2)/ (2*(l^2)) )) + ((sigma_n^2)*kron_delta_func);
end

function k_cap = calc_k_cap(x_matrix, sigma_f,l, sigma_n)
    size_x = length(x_matrix);
    k_cap = zeros(size_x,size_x);
    for i = 1:size_x 
        for j = 1:size_x
            k_cap(i,j) = calc_covar_ij(x_matrix(i),x_matrix(j), sigma_f,l, sigma_n);
            
        end
    end
    k_cap;
end

function k_star = calc_k_star(x_star, x_matrix, sigma_f,l, sigma_n)
    size_x = length(x_matrix);
    k_star = zeros(1,size_x);
    for i = 1:size_x
        k_star(i) = calc_covar_ij(x_star, x_matrix(i), sigma_f,l, sigma_n);
    end
end

function [dissims, y_for_real] = process_taps(radii_data,ref_diffs_norm,ref_diffs_norm_max_ind)
    y_for_real = [];
    dissims = [];

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
        diss = norm([differences(:,:,1) ;differences(:,:,2)]);

        dissims =[dissims diss];
    end
%     y_for_real
end