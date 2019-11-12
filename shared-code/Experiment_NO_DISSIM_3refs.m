classdef Experiment_NO_DISSIM_3refs < handle
    properties
        data
        actual_locations
        tap_number = 0;
        current_rotation = 0;% clockwise = -ve, anti-clock= +ve, 0 = east
        camera
        robot
        ref_diffs_norm_max_ind
        ref_diffs_norm
        dissim_locations
        taps_disp_mu_preds=[];
    end
    properties (Constant)
      sigma_n_diss = 5;%0.5;%1.94;
    end

    methods
        function init(self)
           temp{1}{1} = [];
           self.data = temp;
        end
        
        function do_tap(self,current_step)
            self.tap_number = self.tap_number +1; %count taps on this line

            % start camera recording
            self.camera.start

            % collect data & store
            tacData = self.robot.recordAction;
            self.data{current_step}{self.tap_number} = tacData; % current step is which radius/line currently on, tap_number is how many taps on this current line
            
            
            % stop camera recording
            self.camera.stop
        end%good
        
%         function x_min  = radius_diss_shift(self,dissims, x_matrix)
%         % Return number that when added to the suggested x values, shifts the
%         % values so that the trough (minima) lines up with 0. Uses a gp to estimate
%         % smooth curve rather than using raw dissim values (gp may need tuning
%         % under different circs e.g. harder taps giving higher dissims).Plot raw
%         % and gp estimates for reference.
% 
%             if size(x_matrix(:,1),1) ~= size(dissims',1)
%                 x_matrix(:,1)
%                 dissims' %#ok<NOPRT>
%                 error("dissims and x_matrix are different lengths")
%             end
%             % Get gp for this specific radius
%             [par, ~, ~] = fminunc(@(mypar)gp_max_log_like(mypar(1), mypar(2), self.sigma_n_diss,...
%                                                                 dissims' , x_matrix(:,1)),...
%                                         [10 1] ,optimoptions('fminunc','Display','off'));
% 
%             sigma_f = par(1);
%             l = par(2);
% 
%             % Get K matrix for this radius
%             k_cap = calc_k_cap(x_matrix(:,1), sigma_f,l, self.sigma_n_diss);
% 
%             % Estimate position over length of radius
%             i = 1;
%             for x_star = -10:0.1:10
%         %         if sum(-20:10 == x_star) == 0
%                     x_stars(i) = x_star; %#ok<AGROW>
% 
%                     % setup covariance matrix stuff
%                     k_star      = calc_k_star(x_star, x_matrix(:,1), sigma_f,l, self.sigma_n_diss);
%                     k_star_star = calc_covar_ij(x_star, x_star, sigma_f,l);
% 
%                     % Estimate y
%                     y_star(i) = k_star * inv(k_cap) * dissims'; %#ok<MINV,AGROW>
% 
%                     % Estimate variance
%                     var_y_star(i) = k_star_star - (k_star * inv(k_cap) * transpose(k_star)); %#ok<MINV,AGROW>
%                     if var_y_star(i) < 0.0000
%                         var_y_star(i) =0; %#ok<AGROW> % otherwise -0.0000 causes errors with sqrt()
%                     end
% 
%                     i = i+1;
%         %         end
%             end
% 
%             [~,x_min_ind] = min(y_star);
%             x_min = -x_stars(x_min_ind);
% % 
% %             %% Plot input
% %             figure(2)
% %             subplot(1,2,1)
% %         %     clf
% %             hold on
% %             title("Original")
% %         %     fill([x_stars, fliplr(x_stars)],...
% %         %          [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
% %         %          [1 1 0.8])
% % 
% %             plot(x_matrix(:,1), dissims, '+')
% %             plot(x_matrix(:,1), dissims)
% %             plot(x_stars, y_star)
% %             axis([-10 10 0 90])
% %             hold off
% % 
% %             %% Plot output
% %             subplot(1,2,2)
% %             hold on
% %             title("Troughs aligned")
% %             plot(x_matrix(:,1)+x_min, dissims, '+')
% %             plot(x_stars+x_min, y_star)
% %             grid on
% %             grid minor
% %             axis([-10 10 0 90])
% %             hold off
% 
%         end%good
        
        function move_to_location(self,location)
        % Move to location, location = [x y theta] . x and y are in mm, theta is in
        % degrees. Theta should be between -180 and 180, and in absolute terms, not
        % relative turning. Theta will be saved as current_rotation, and so
        % current_rotation should NOT be changed outside this function!
        % Give x in global coords, will be negated to move properly in
        % workframe coords.

            if ~isequal(size(location), [1 3])
                location %#ok<NOPRT>
                error("Move command is of wrong dimensions, should be [x y theta]")
            end

            if abs(location(3)) > 180
                location(3)
                error("Angle to turn to is not within -180 to 180 deg")
            end            
            
            % check x,y are within boundaries
            if location(1) < -150 || location(1) > 50 || location(2) < -100 ||location(2) > 100
                error("location is outside of safe box, aborting")
            end

            amount_to_turn = location(3) - self.current_rotation;
            
            % move in steps of no more than 90deg in either direction (to
            % prevent wrapping wire around too far - robot direction is 
            % unpredictable at&over 180deg).
            while amount_to_turn > 90
                self.robot.move([-location(1) location(2) 0 0 0 -(self.current_rotation+90)])
                self.current_rotation = self.current_rotation+90;
                amount_to_turn = location(3) - self.current_rotation;
            end
            while amount_to_turn < -90
                self.robot.move([-location(1) location(2) 0 0 0 -(self.current_rotation-90)])
                self.current_rotation = self.current_rotation-90;
                amount_to_turn = location(3) - self.current_rotation;
            end
            self.robot.move([-location(1) location(2) 0 0 0 -location(3)]) %NB toolframe is oposite to world
            self.current_rotation = location(3); % update what the current location is
        end%good
        
        function move_and_tap(self,location,current_step)
            disp(strcat("Moving to ",mat2str(location),' and tapping'))
            self.move_to_location(location)
            self.do_tap(current_step)
            self.actual_locations{current_step}{self.tap_number} = location;
        end%good
        
        function [model, current_step] = bootstrap(self)
        % Find initial search line angle, collect data along this orientation, find
        % edge, add data to model with corrected x disps. 
            disp("Starting bootstrap...")
        
            current_step = 1; % bootstrap is always the first set of collected data
            model = GPLVM_NO_DISSIM;

            self.current_rotation = 0;% clockwise = -ve, anti-clock= +ve
            
            px = -3;
            py = 0;
            % find direction of first line,
            %3 taps, at 90degs, calc gradient (gradient decent?)
            self.move_and_tap([px py self.current_rotation],current_step);
            [~, dissim_o]= self.process_single_tap(self.data{current_step}{self.tap_number})

            self.move_and_tap([px+1 py self.current_rotation],current_step);
            [~, dissim_x]= self.process_single_tap(self.data{current_step}{self.tap_number})

            self.move_and_tap([px py+1 self.current_rotation],current_step);
            [~, dissim_y]= self.process_single_tap(self.data{current_step}{self.tap_number})

            % find angle where direction is most decreasing in dissimilarity
            rotation_offset = atan2d(dissim_o - dissim_y, dissim_o - dissim_x)

%             self.current_rotation = rotation_offset; %assumes ref tap is to west of tip and stimulus is near west for first taps (as 0deg is east)

            % collect line
            %%%%%%%%%%%%%%%%%%%%%%REPEATED CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            n_useless_taps = self.tap_number; %so can exlude points later on
            % tap along "radius" 
            for disp_from_start = -10:5:10 
                temp_point = [px py] + disp_from_start*[cosd(rotation_offset)...
                                                        sind(rotation_offset)];
                self.move_and_tap([temp_point rotation_offset],current_step);%hereafter current_rotation == rotation_offset
            end
            
            %%%Start of changes
            % calc dissim, align to 0 (edge)
            ys_for_real = self.process_taps(self.data{current_step});
            xs_default = [-10:5:10]';
            
            % init model hyper params using collected line data with defualt (false) shift
            model.set_new_hyper_params(ys_for_real(n_useless_taps+1:end,:), [xs_default ones(length(xs_default),1)])
            
            % Add single ref tap as first data in model
            ref_ys = [self.ref_diffs_norm{1}(self.ref_diffs_norm_max_ind{1} ,:  ,1) self.ref_diffs_norm{1}(self.ref_diffs_norm_max_ind{1} ,:  ,2);...
                      self.ref_diffs_norm{2}(self.ref_diffs_norm_max_ind{2} ,:  ,1) self.ref_diffs_norm{2}(self.ref_diffs_norm_max_ind{2} ,:  ,2);...
                      self.ref_diffs_norm{3}(self.ref_diffs_norm_max_ind{3} ,:  ,1) self.ref_diffs_norm{3}(self.ref_diffs_norm_max_ind{3} ,:  ,2)];
            refs_xs = [-1 1; 0 1; 1 1];
            model.add_data_to_model_direct(ref_ys, refs_xs)
            
            %find shift using gplvm and single ref, previous lines and new line
            x_min  = model.radius_diss_shift(ys_for_real(n_useless_taps+1:end,:), xs_default);%remove first 3 y points as not in line
            xs_current_step = xs_default + x_min; % so all minima are aligned
            
            %add data to model by optimising mu for line
            model.add_a_radius(ys_for_real(n_useless_taps+1:end,:), xs_current_step)
            
            %%%End of changes
            
            % location closest to 0 dissim is point for next extrapolation
            self.dissim_locations = [px py] - x_min*[cosd(self.current_rotation)...
                                                     sind(self.current_rotation)]; %TODO check sign of x_min %cat onto bottom, use dissim_locations(end,:) to get last point

            
            
            %%%%%%%%%%%%%%%%%%%%%%REPEATED CODE END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            disp("...finished bootstrap")
        end%one TODO
        
        function [processed_tap, diss] = process_single_tap(self,tap_data)
        % Tap data should be (1:n_frames,1:127,1:2) dimensions
        % processed_tap is (1:254)
        % diss is (1:1)

            current_tap_data_norm = tap_data(: ,:  ,:)- tap_data(1 ,:  ,:);

            % diff_between_noncontacts = ref_tap(1 ,:  ,:) - radii_data{1,tap_num}(1 ,:  ,:); %TODO throw error if too large?

            n_pins = size(tap_data,2);
            max_i = zeros(1, n_pins);
            for pin = 1:n_pins

                [~,max_ind_x]=max(abs(current_tap_data_norm(: ,pin  ,1)));
                [~,max_ind_y]=max(abs(current_tap_data_norm(: ,pin  ,2)));

                max_i(1,pin) = max_ind_x;
                max_i(2,pin) = max_ind_y;
            end

            average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2)); % want to compare same frame across tap, not different frames for each pin

            processed_tap = [current_tap_data_norm(average_max_i,:,1) current_tap_data_norm(average_max_i,:,2)];

            % dissimilarity measure 
            differences = self.ref_diffs_norm{2}(self.ref_diffs_norm_max_ind{2} ,:  ,:) ...
                                  - current_tap_data_norm(average_max_i,:,:);
            diss = norm([differences(:,:,1)'; differences(:,:,2)']);

        end%good
        
        function [y_processed] = process_taps(self,radii_data)
        % Return modified y data (1 by 256), dissimilarity data and the minimum x point
        % for a single radius of data (intended to be a single radius).

            y_processed = [];
%             dissims = [];

            for tap_num = 1:length(radii_data) %TODO? is length safe, maybe use size(,)?
                current_tap_data_norm = radii_data{1,tap_num}(: ,:  ,:)- radii_data{1,tap_num}(1 ,:  ,:);
                
                n_pins = size(radii_data{1,1},2);
                max_i = zeros(1, n_pins);
                for pin = 1:n_pins

                    [~,max_ind_x]=max(abs(current_tap_data_norm(: ,pin  ,1)));
                    [~,max_ind_y]=max(abs(current_tap_data_norm(: ,pin  ,2)));

                    max_i(1,pin) = max_ind_x;
                    max_i(2,pin) = max_ind_y;
                end

                average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2)); % want to compare same frame across tap, not different frames for each pin

%                 differences = self.ref_diffs_norm(self.ref_diffs_norm_max_ind ,:  ,:) ...
%                               - current_tap_data_norm(average_max_i,:,:); 

                y_processed = [y_processed;...
                              current_tap_data_norm(average_max_i,:,1) current_tap_data_norm(average_max_i,:,2)]; %#ok<AGROW>

%                 diss = norm([differences(:,:,1)'; differences(:,:,2)']);
%                 dissims =[dissims diss]; %#ok<AGROW>
            end
        end%good
        
    end%methods
    
end%class