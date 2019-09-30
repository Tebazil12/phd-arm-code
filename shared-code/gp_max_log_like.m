function val = gp_max_log_like(sigma_f, l, sigma_n, y, x_matrix)
%     sigma_f
%     l
%     sigma_n
    k_cap = calc_k_cap(x_matrix, sigma_f, l, sigma_n);
%     
%     if det(k_cap) == 0 
%         error("Determinant was 0, unable to continue")
%     end
%     
%     part1 = -0.5 * y' * pinv(k_cap) * y;
%     part2 = -0.5 * log(det(k_cap)); % NB, |K| means det(K), not abs(K) or norm(K)
%     part3 = -0.5 * length(y)*log(2*pi);
%     neg_val = part1 + part2 + part3;
%     val = - neg_val; % because trying to find max with a min search

    [n, ~] = size(k_cap);
    R = chol(k_cap); % K = R'*R
    
    % Calculate the log determinant of the covariance matrix
%     logdet_K = sum(2*log(R(1:n+1:end)))
%     logdet_K = det(R)
    logdet_K = sum(log(diag(R)));
    diag(R);


    % Calculate the log likelihood
    alpha = R \ (R' \ y);
    
%     val = 0.5*mean(sum(y'.*alpha, 1)) + 0.5*logdet_K + 0.5*n*log(2*pi);
    val = (0.5*y'*alpha) + 0.5*logdet_K + 0.5*n*log(2*pi);
%     val = (0.5* y' * (R' * R)\y )+ 0.5*logdet_K + 0.5*n*log(2*pi);



end