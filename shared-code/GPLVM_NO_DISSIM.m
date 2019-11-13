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
            
%             self.y_gplvm_input_train = y_input;
%             self.x_gplvm_input_train = x_input;
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
        
        function add_data_to_model_direct(self, ys, xs)
        % note, currently REPLACES all model data with ys and xs, it does
        % not CAT ys and xs
            self.y_gplvm_input_train = ys;
            self.x_gplvm_input_train = xs;            
        end
        
        function estimated_shift  = radius_diss_shift(self, ys_line, xs_line, ref_y, ref_x)
        % Predicts shift of data using gplvm, copying
        % code from ...optmmu_sep.m. Replaces funciton that was in Experiment
        
            disp("Getting line shift")

            %%%
                
            init_latent_vars = 0;
            %% %%%%%%%%%%%%% Training Shift %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
            
            [par, ~, flag] = fminunc(@(opt_pars)gplvm_max_log_like(self.sigma_f,...
                                                                   [self.l_disp self.l_mu],...
                                                                   self.sigma_n_y,...
                                                                   [ref_y; self.y_gplvm_input_train; ys_line],...
                                                                   [ref_x; self.x_gplvm_input_train; xs_line+opt_pars(1) ones(size(xs_line))]),...
                                        init_latent_vars,...
                                        optimoptions('fminunc','Display','off','MaxFunEvals',10000));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            flag
            if flag < 1
                warning("fminsearch was not happy")
                flag
            end 
            
            estimated_shift = par(1)
            
            if abs(estimated_shift) > 10
                estimated_shift
                warning('Estimated shift of line is greater than +-10mm')
            elseif abs(estimated_shift) == 10
                warning('Shift indicates edge is at far end of line, therefore may be inaccurate')
            end
            
            if estimated_shift > 20
                estimated_shift = 20;
                warning('Shift too large, setting to 20mm')
            elseif estimated_shift < -20
                estimated_shift = -20;
                warning('Shift too (negatively) large, setting to -20mm')
            end

%             %% %%%%%%%%%%%%% Training 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
%             init_hyper_pars_3 = [0];
% 
%             [par, ~, flag] = fminunc(@(opt_pars)gplvm_max_log_like(sigma_f, ...
%                                                                       [l_disp l_mu], ...
%                                                                       sigma_n_y,...
%                                                                       [y_ref_taps; y_gplvm_input_train] ,...
%                                                                       [[disp_ref_taps; disp_gplvm_input_train; x_real(MIN_I_TRAIN:MAX_I_TRAIN,line)+real_shift+estimated_shift] ...
%                                                                        [mu_ref_taps; mu_gplvm_input_train; ones(21,1)*opt_pars(1)]]),...
%                                         init_hyper_pars_3,...
%                                         optimoptions('fminunc','Display','off','MaxFunEvals',10000));
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             flag
%             if flag < 1
%                 warning("fminsearch was not happy")
%                 flag
%             end 
% 
%             par
% 
%             estiamted_mu = par(1)
%             mu_error_train= estiamted_mu - (training_angle_indexes(line)-10)/4.5
%             % par(4)
%             % par(5)
% 
%             disp_gplvm_input_train = [disp_gplvm_input_train;...    
%                                       x_real(MIN_I_TRAIN:MAX_I_TRAIN,line)+real_shift+estimated_shift];
%             mu_gplvm_input_train = [mu_gplvm_input_train;...
%                                         ones(21,1)*estiamted_mu];
% 
%             shifts = [shifts; estimated_shift];
%             mu_error_trains = [mu_error_trains mu_error_train];
%             
            %%%
%             init_latent_vars = 0;
%             %%%%%%%%%%%%%% Radius Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %note, mu for ref tap and line should be same to maximise disp
%             %alignement
%             [par, ~, flag] = fminunc(@(opt_pars)gplvm_max_log_like(self.sigma_f,...
%                                                                    [self.l_disp self.l_mu],...
%                                                                    self.sigma_n_y,...
%                                                                    [self.y_gplvm_input_train; ys_for_real],...
%                                                                    [self.x_gplvm_input_train; xs_current_step+opt_pars(1) ones(size(xs_current_step))]),...
%                                         init_latent_vars,...
%                                         optimoptions('fminunc','Display','off'));
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             if flag < 1
%                 flag %#ok<NOPRT>
%             end 
% 
%             new_mu = par; %TODO add error checking on mu, like if mu is really far from rest of graph, reorder graph or something?
%             
%             self.y_gplvm_input_train = [self.y_gplvm_input_train;...
%                                         ys_for_real];
%             self.x_gplvm_input_train = [self.x_gplvm_input_train;...
%                                         xs_current_step ones(size(xs_current_step))*new_mu];
        end
    end
    
end