%% Load data from repeated single radii exp.
% x will be displacement (later also angle/depth)
% y will be normalized tap at max disp (raw data minus first frame, max
% disps of those).

clearvars
clf

load('/home/lizzie/git/masters-tactile/data/singleRadius2018-11-19_1524/c01_01.mat')
all_data{1}= data;

x_real = [10:-1:-20]';

%% Reference tap stuff
ref_tap = all_data{1}{1,11}(:,:,:); 

% Normalize data, so get distance moved not just relative position
ref_diffs_norm = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); %normalized, assumes starts on no contact/all start in same position

% find the frame in ref_diffs_norm with greatest diffs
[~,an_index] = max(abs(ref_diffs_norm));
ref_diffs_norm_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));

%% Filter raw data (normalize, find max frame)
y_for_real = [];
dissims = [];
num =[];

for tap_num = 1:31
    current_tap_data_norm = all_data{1}{1,tap_num}(: ,:  ,:)...
                            - all_data{1}{1,tap_num}(1 ,:  ,:);

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

    y_for_real = [y_for_real; differences(:,:,1) differences(:,:,2)];
    diss = norm([differences(:,:,1) ;differences(:,:,2)]);
    num = [num (31-(tap_num))-20];

    dissims =[dissims diss];
end
y_for_real



%% Optimize hyper-params

init_hyper_pars = [0.5 0.2 0.2];
init_latent_vars = [x_real];

[par, fval, flag] = fminunc(@(opt_pars)gplvm_max_log_like(opt_pars(1), opt_pars(2), opt_pars(3),...
                                                          y_for_real , [opt_pars(4:end)']),...
                            [init_hyper_pars init_latent_vars'] )

if flag ~= 1
    warning("fminsearch was not happy")
    flag
end 

if round(par(1),1) == 0 || round(par(2),1) == 0 || round(par(3),1) == 0
    warning("A hyper-parameter is zero! Bad fit")
end

sigma_f = par(1);
l = par(2);
sigma_n = par(3);
predicted_x = [par(4:end)];
x_points = [predicted_x'];

%% Validate params over many points
k_cap = calc_k_cap(x_points, sigma_f,l, sigma_n);

i = 1;
for x_star = -20:0.5:10
    x_stars(i) = x_star;
    
    % setup covariance matrix stuff
    k_star      = calc_k_star(x_star, x_points, sigma_f,l, sigma_n);
    k_star_star = calc_covar_ij(x_star, x_star, sigma_f,l, sigma_n);
    
    % Estimate y
    k_star * inv(k_cap) * dissims'
    y_star(i) = k_star * inv(k_cap) * dissims';
    
    % Estimate variance
    var_y_star(i) = k_star_star - (k_star * inv(k_cap) * transpose(k_star));
    if var_y_star(i) < 0.0000
        var_y_star(i) =0; % otherwise -0.0000 causes errors with sqrt()
    end
    
    i = i+1;
end

%% Plot everything

figure(1)
title("All angles")
hold on
xlabel("Displacemt / mm")
ylabel("dissim")
axis([-20 10 0 40])

% plot Stdev band
fill([x_stars, fliplr(x_stars)],...
     [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
     [1 1 0.8],...
     'EdgeColor','none')

% plot predicted data
plot(x_points,dissims, 'b') 
scatter(x_points,dissims, 'b', '+') 

% plot predictions from interpolating
plot(x_stars,y_star, 'r') %TODO how does this even work?!

% plot actual data
plot(num, dissims, 'k')
scatter(num, dissims, 'k','+')

hold off

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



