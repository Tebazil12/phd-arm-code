function val = gplvm_max_log_like(sigma_f, l, sigma_n, y, x_matrix)
%     sigma_f
%     l
%     sigma_n
    x_matrix;
    
%     if sum(x_matrix(end,1) == x_matrix(:,1)) > 1
% %         warning("repetition of x exactly is not allowed, adding 0.0000000000001")
%         x_matrix(end,1) = x_matrix(end,1) +0.0000000000001;
%     end
%     if sum(x_matrix(end,2) == x_matrix(:,2)) > 1
% %         warning("repetition of x exactly is not allowed, adding 0.0000000000001")
%         x_matrix(end,2) = x_matrix(end,2) +0.0000000000001;
%     end

    % Having exact same x combos causes errors as det(K) = 0 so can't be
    % inverted, which is required for later
%     if sum(sum(x_matrix(end,:) == x_matrix,2) == 2) >1
%         warning("repetition of x exactly is not allowed, adding 0.0000000000001")
%         x_matrix(end) = x_matrix(end) +0.0000000000001;
%     end

    [d_cap, n_cap] = size(y);
    s_cap = (1/d_cap) * (y * y'); 
    
%     x_matrix(1,:)
    k_cap = calc_k_cap(x_matrix, sigma_f, l, sigma_n);
    
    R = chol(k_cap);
    logdet_K = sum(diag(R));
    
    if det(k_cap) == 0 
        error("Determinant was 0, unable to continue")
    end
    
    part1 = - (d_cap * n_cap) * 0.5 * log(2 * pi);
%     part2 = - (d_cap) * 0.5 * log(det(k_cap)); % NB, |K| means det(K), not abs(K) or norm(K)
    part2 = - (d_cap) * 0.5 * logdet_K;
    part3 = - (d_cap) * 0.5 * trace(pinv(k_cap) * s_cap);
    
    neg_val = part1 + part2 + part3;
    val = - neg_val; % because trying to find max with a min search
end