% Define covariance function, k:
function k = calc_covar_ij(x,x_prime,sigma_f,l)
    if length(sigma_f) ~= 1
        error("sigma_f must be a scalar, not a matrix")
    end

    x;
    x_prime;
    x_diff = x - x_prime;
%     if sum(x_diff ~= 0) == 0
%         warning("x_diff is 0, cannot devide by 0")
%     end
    x_diff_sqr = x_diff.^2;
    x_sqr_l_sqr = (x_diff_sqr) ./ (2* (l.^2));
    for dimesions_of_x = 1:length(x_diff)
        if isnan(x_sqr_l_sqr(dimesions_of_x))
            x_sqr_l_sqr(dimesions_of_x) = 0;
        end
    end
    sum_sqrs = sum(x_sqr_l_sqr);
    exp_x_diff = exp(-sum_sqrs); % edit l to be different for disp and "angle"
    k = (exp_x_diff.* (sigma_f^2) );
    if isnan(k)
        error("cov trying to return nan")
    end
%     k_sum = sum(k_matrices);
%     k = k_sum;
    
%     x;
%     x_prime;
%     x_diff = x - x_prime;
%     x_diff_sqr = x_diff.^2;
%     exp_x_diff = exp(-(x_diff_sqr) / (2* (l^2))); % edit l to be different for disp and "angle"
%     k_matrices = (exp_x_diff.* (sigma_f^2) );%  +  ((sigma_n^2)*kron_delta_func);
%     k_sum = sum(k_matrices);
%     k = k_sum;
    
    
%     if length(x) > 1 || length(x_prime) > 1
%         error("Somehow x's are greater than 1D, cov() can't cope")
%     end
%     
%     k = (sigma_f^2 * exp( (-(x - x_prime)^2)/ (2*(l^2)) )) + ((sigma_n^2)*kron_delta_func); %TODO check if this noise is broken for actaul things like this
end