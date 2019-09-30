function k_star = calc_k_star(x_star, x_matrix, sigma_f,l, sigma_n)
    if size(x_matrix,2) ~= size(x_star,2)
        error("Dimensions of x_star and x_matrix are not the same")
    end
    size_x = length(x_matrix);
    k_star = zeros(1,size_x);
    for i = 1:size_x
        k_star(i) = calc_covar_ij(x_star, x_matrix(i), sigma_f,l);
        if isnan(k_star(i))
            error("Part of k_star is nan")
        end
    end
end