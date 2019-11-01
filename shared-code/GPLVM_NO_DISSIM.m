% Class handling the creation, updating and use of a GP-LVM model.
% Copyright (C) 2019  Elizabeth A. Stone
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <https://www.gnu.org/licenses/>.

classdef GPLVM_NO_DISSIM < handle
    properties
        sigma_f
        l_disp 
        l_mu
        
        y_gplvm_input_train
        x_gplvm_input_train
    end
    properties (Constant)
      sigma_n_y = 1.14;
    end

    methods       
        
        function set_new_hyper_params(self, y_input, x_input)
        % Use y_input and x_input to optimize the hyper parameters sigma_f,
        % l_disp and l_mu, saving these and the input in self.
            if  size(x_input,2) ~= 2
                size(y_input)
                size(x_input) 
                error("input(s) to hyper param optimization are not correct dimension(s)")
            elseif size(y_input,1) ~= size(x_input,1)
                error("y_inout and x_input do not have the same number of entries")
            end
            
            %% Optimize hyper-params %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
            init_hyper_pars = [1 300 5];

            [par, ~, flag] = fminunc(@(opt_pars)gplvm_max_log_like(opt_pars(1), ...
                                                                   [opt_pars(2) opt_pars(3)], ...
                                                                   self.sigma_n_y,...
                                                                   y_input, x_input),...
                                     init_hyper_pars,...
                                     optimoptions('fminunc','Display','off','MaxFunEvals',10000));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if flag < 1
                flag %#ok<NOPRT>
                warning("fminsearch was not happy")   
            end 

            if round(par(1),1) == 0 || round(par(2),1) == 0
                warning("A hyper-parameter is zero! Probably a bad fit")
            end
            
            self.sigma_f = par(1);
            self.l_disp = par(2);
            self.l_mu = par(3);
            
            self.y_gplvm_input_train = y_input;
            self.x_gplvm_input_train = x_input;
        end%good (just niceties to add)
        
        function [pred_x, pred_mu] = predict_singletap(self, pred_tap)
            init_latent_vars = [0 0];

            %%%%%%%%%%%%%% GPLVM Point Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [par, ~, flag] = fminunc(@(opt_pars)gplvm_max_log_like(self.sigma_f,...
                                                                   [self.l_disp self.l_mu],...
                                                                   self.sigma_n_y,...
                                                                   [self.y_gplvm_input_train; pred_tap],...
                                                                   [self.x_gplvm_input_train; opt_pars(1) opt_pars(2)]),...
                                     init_latent_vars,...
                                     optimoptions('fminunc','Display','off'));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if flag < 1
                flag %#ok<NOPRT>
                warning("Bad flag from optimizing single tap")
            end 

            pred_x  = par(1);
            pred_mu = par(2);
        end
        
        function add_a_radius(self, ys_for_real, xs_current_step)
        % Predicts mu for the given radius and adds these to the x and y
        % inputs for the gplvm model. 
            disp("Adding a line to model")
            init_latent_vars = 0;
            %%%%%%%%%%%%%% Radius Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [par, ~, flag] = fminunc(@(opt_pars)gplvm_max_log_like(self.sigma_f,...
                                                                   [self.l_disp self.l_mu],...
                                                                   self.sigma_n_y,...
                                                                   [self.y_gplvm_input_train; ys_for_real],...
                                                                   [self.x_gplvm_input_train; xs_current_step ones(size(xs_current_step))*opt_pars(1)]),...
                                        init_latent_vars,...
                                        optimoptions('fminunc','Display','off'));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if flag < 1
                flag %#ok<NOPRT>
            end 

            new_mu = par; %TODO add error checking on mu, like if mu is really far from rest of graph, reorder graph or something?
            
            self.y_gplvm_input_train = [self.y_gplvm_input_train;...
                                        ys_for_real];
            self.x_gplvm_input_train = [self.x_gplvm_input_train;...
                                        xs_current_step ones(size(xs_current_step))*new_mu];
        end
        
    end
    
end