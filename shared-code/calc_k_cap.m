function k_cap = calc_k_cap(x_matrix, sigma_f,l, sigma_n)
    size_x = length(x_matrix);
    k_cap = zeros(size_x,size_x);
    for i = 1:size_x 
        for j = 1:size_x
            k_cap(i,j) = calc_covar_ij(x_matrix(i,:),x_matrix(j,:), sigma_f,l);
            
        end
    end
    k_cap =k_cap + eye(size(k_cap))*sigma_n ;
end