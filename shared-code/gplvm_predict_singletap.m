init_latent_vars = [0 0];

%%%%%%%%%%%%%% GPLVM Point Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[par, ~, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f,...
                                                       [l_disp l_mu],...
                                                       sigma_n_y,...
                                                       [y_gplvm_input_train; new_tap],...
                                                       [x_gplvm_input_train; opt_pars(1) opt_pars(2)]),...
                         init_latent_vars,...
                         optimoptions('fminunc','Display','off'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if flag < 1
    flag
    warning("Bad flag from optimizing single tap")
end 

new_x  = par(1);
new_mu = par(2); %TODO do we use mu from here?    
    
