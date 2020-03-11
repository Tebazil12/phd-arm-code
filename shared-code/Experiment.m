classdef Experiment < handle
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
        
        robot_serial
        ref_tap
        still_tap
        still_tap_array
        sensor
        search_angles
    end
    properties (Constant)
      sigma_n_diss = 5;%5;%0.5;%1.94;
    end

    methods
        function init(self)
           temp{1}{1} = [];
           self.data = temp;
        end
        
%         function do_tap(self,current_step)
%             self.tap_number = self.tap_number +1; %count taps on this line
% 
%             % start camera recording
%             self.camera.start
% 
%             % collect data & store
%             tacData = self.robot.recordAction;
%             self.data{current_step}{self.tap_number} = tacData; % current step is which radius/line currently on, tap_number is how many taps on this current line
%             
%             
%             % stop camera recording
%             self.camera.stop
%         end%good
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
%                                         [1 1] ,optimoptions('fminunc','Display','off'));
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
%             hold on
%             plot(x_matrix, dissims)
%             plot(x_stars, y_star)
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

        function x_min  = radius_diss_shift(self,dissims, x_matrix)
        % Return number that when added to the suggested x values, shifts the
        % values so that the trough (minima) lines up with 0. Uses a gp to estimate
        % smooth curve rather than using raw dissim values (gp may need tuning
        % under different circs e.g. harder taps giving higher dissims).Plot raw
        % and gp estimates for reference.

            if size(x_matrix(:,1),1) ~= size(dissims',1)
                x_matrix(:,1)
                dissims' %#ok<NOPRT>
                error("dissims and x_matrix are different lengths")
            end
            % Get gp for this specific radius
            [par, ~, ~] = fminunc(@(mypar)gp_max_log_like(mypar(1), mypar(2), self.sigma_n_diss,...
                                                                dissims' , x_matrix(:,1)),...
                                        [1 1] ,optimoptions('fminunc','Display','off'));

            sigma_f = par(1);
            l = par(2);

            % Get K matrix for this radius
            k_cap = calc_k_cap(x_matrix(:,1), sigma_f,l, self.sigma_n_diss);

            % Estimate position over length of radius
            i = 1;
            for x_star = x_matrix(1):0.1:x_matrix(end)
        %         if sum(-20:10 == x_star) == 0
                    x_stars(i) = x_star; %#ok<AGROW>

                    % setup covariance matrix stuff
                    k_star      = calc_k_star(x_star, x_matrix(:,1), sigma_f,l, self.sigma_n_diss);
                    k_star_star = calc_covar_ij(x_star, x_star, sigma_f,l);

                    % Estimate y
                    y_star(i) = k_star * inv(k_cap) * dissims'; %#ok<MINV,AGROW>

                    % Estimate variance
                    var_y_star(i) = k_star_star - (k_star * inv(k_cap) * transpose(k_star)); %#ok<MINV,AGROW>
                    if var_y_star(i) < 0.0000
                        var_y_star(i) =0; %#ok<AGROW> % otherwise -0.0000 causes errors with sqrt()
                    end

                    i = i+1;
        %         end
            end

            [~,x_min_ind] = min(y_star);
            x_min = -x_stars(x_min_ind);
% 
%             %% Plot input
%             figure(2)
%             subplot(1,2,1)
%         %     clf
%             hold on
%             title("Original")
%         %     fill([x_stars, fliplr(x_stars)],...
%         %          [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
%         %          [1 1 0.8])
% 
%             plot(x_matrix(:,1), dissims, '+')
            hold on
            plot(x_matrix, dissims)
            plot(x_stars, y_star)
%             axis([-10 10 0 90])
%             hold off
% 
%             %% Plot output
%             subplot(1,2,2)
%             hold on
%             title("Troughs aligned")
%             plot(x_matrix(:,1)+x_min, dissims, '+')
%             plot(x_stars+x_min, y_star)
%             grid on
%             grid minor
%             axis([-10 10 0 90])
%             hold off

        end%good
        
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
        
        function [model, current_step] = bootstrap(self, EDGE_TRACK_DISTANCE)
        % Find initial search line angle, collect data along this orientation, find
        % edge, add data to model with corrected x disps. 
            disp("Starting bootstrap...")
        
            current_step = 1; % bootstrap is always the first set of collected data
            model = GPLVM;

            % collect line
            %%%%%%%%%%%%%%%%%%%%%%REPEATED CODE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            n_useless_taps = self.tap_number; %so can exlude points later on
            
            resp = writeread(self.robot_serial,"FR_leg_forward_hover") %so tip don't brake
            pause(1.5);
            
            self.search_angles = -5:1:25 ;
            % tap along "radius" 
            for disp_from_start = self.search_angles+EDGE_TRACK_DISTANCE

                % move distance predicted 
                if disp_from_start < 0 
                    command_to_send = "-";
                else
                    command_to_send = "+";
                end

                if disp_from_start <10 && disp_from_start >-10
                    command_to_send = strcat(command_to_send, "0");
                end

                command_to_send = strcat(command_to_send, int2str(abs(disp_from_start)), "_FR_rotateHip")

                % Do tap
                resp = writeread(self.robot_serial,command_to_send)%this is a tap
%                 self.sensor.setPins(self.still_tap_array);
                pause(1.5); % give time to get there
                self.tap_number = self.tap_number +1;
                pins = self.sensor.record;
                self.data{current_step}{self.tap_number} = pins;
                if size(pins,2) ~= 37
                    error("New tap is not same size as ref_tap")
                end
            end

            [dissims, ys_for_real] = self.process_taps(self.data{current_step});
            xs_default = self.search_angles';
            x_min  = self.radius_diss_shift(dissims(n_useless_taps+1:end), xs_default);%remove first 3 points as not in line

            xs_current_step = xs_default + x_min; % so all minima are aligned

            %error check, see if minima was actually in range (ie end points arent minima, but somewhere in middle)
            [~,min_i] = min(dissims(n_useless_taps+1:end));
            if  min_i== 1 || min_i == length(dissims(n_useless_taps+1:end))
                warning("Minimum diss was at far end, actual minima probably not found, model may be bad")
            end

            %%%%%%%%%%%%%%%%%%%%%%REPEATED CODE END%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % init model hyper params using collected line data
            model.set_new_hyper_params(ys_for_real(n_useless_taps+1:end,:), [xs_current_step ones(length(self.search_angles),1)])
            
            % rotate hips by x_min in next phases of walking
            turn_hips_by = round(-x_min)
            
            %%%% MORE REPEATED CODE %%%%%%
            if abs(turn_hips_by) < 2 % not worth time & energy twisting if less than 2
                start_hip_rotation_command = "+00";
                start_hip_antirotation_command= "+00";
            else
                if turn_hips_by < 0 
                    start_hip_rotation_command = "-";
                    start_hip_antirotation_command = "+";
                else
                    start_hip_rotation_command = "+";
                    start_hip_antirotation_command = "-";
                end

                if turn_hips_by <10 && turn_hips_by >-10
                    start_hip_rotation_command = strcat(start_hip_rotation_command, "0");
                    start_hip_antirotation_command = strcat(start_hip_antirotation_command, "0");
                end

                start_hip_rotation_command = strcat(start_hip_rotation_command, int2str(abs(turn_hips_by)))
                start_hip_antirotation_command = strcat(start_hip_antirotation_command, int2str(abs(turn_hips_by)));
            end 

             % next walking steps ...
        %     resp = writeread(self.robot_serial,"FR_leg_side")
        %     pause(1.5);
            command_to_send = strcat(start_hip_rotation_command, "_FR_rotateHip")
            resp = writeread(self.robot_serial,command_to_send)
            pause(3);

            command_to_send = strcat(start_hip_antirotation_command, "_BLm_rotateHip");
            resp = writeread(self.robot_serial,command_to_send)
            pause(3);

            command_to_send = strcat(start_hip_antirotation_command, "_BR_rotateHip");
            resp = writeread(self.robot_serial,command_to_send)
            pause(3);

            command_to_send = strcat(start_hip_antirotation_command, "_FLm_rotateHip");
            resp = writeread(self.robot_serial,command_to_send)
            pause(3);

%             resp = writeread(self.robot_serial,"FR_leg_forward_tap") % this has to be here as turning for tapping needs to happen in forward pose, so can't move from side to back in hip twist
%             pause(1.5);

            resp = writeread(self.robot_serial,"FRf_body_forward")
            pause(1.5);

            resp = writeread(self.robot_serial,"BL_leg_forward")
            pause(1.5);

            command_to_send = strcat(start_hip_rotation_command, "_BLs_rotateHip");
            resp = writeread(self.robot_serial,command_to_send)
            pause(1.5);

            resp = writeread(self.robot_serial,"FL_leg_forward")
            pause(1.5);

            command_to_send = strcat(start_hip_rotation_command, "_FLe_rotateHip");
            resp = writeread(self.robot_serial,command_to_send)
            pause(1.5);

            resp = writeread(self.robot_serial,"FLf_body_forward")
            pause(1.5);

            resp = writeread(self.robot_serial,"BR_leg_forward")
            pause(1.5);

            
            %%%% END - MORE REPEATED CODE %%%%%%
            
            disp("...finished bootstrap")
        end%one TODO
        
%         function [processed_tap, diss] = process_single_tap(self,tap_data)
%         % Tap data should be (1:n_frames,1:127,1:2) dimensions
%         % processed_tap is (1:254)
%         % diss is (1:1)
% 
%             current_tap_data_norm = tap_data(: ,:  ,:)- tap_data(1 ,:  ,:);
% 
%             % diff_between_noncontacts = ref_tap(1 ,:  ,:) - radii_data{1,tap_num}(1 ,:  ,:); %TODO throw error if too large?
% 
%             n_pins = size(tap_data,2);
%             max_i = zeros(1, n_pins);
%             for pin = 1:n_pins
% 
%                 [~,max_ind_x]=max(abs(current_tap_data_norm(: ,pin  ,1)));
%                 [~,max_ind_y]=max(abs(current_tap_data_norm(: ,pin  ,2)));
% 
%                 max_i(1,pin) = max_ind_x;
%                 max_i(2,pin) = max_ind_y;
%             end
% 
%             average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2)); % want to compare same frame across tap, not different frames for each pin
% 
%             processed_tap = [current_tap_data_norm(average_max_i,:,1) current_tap_data_norm(average_max_i,:,2)];
% 
%             % dissimilarity measure 
%             differences = self.ref_diffs_norm(self.ref_diffs_norm_max_ind ,:  ,:) ...
%                                   - current_tap_data_norm(average_max_i,:,:);
%             diss = norm([differences(:,:,1)'; differences(:,:,2)']);
% 
%         end%good
        function [processed_tap, diss] = process_single_tap(self,tap_data)
        % Tap data should be (1:n_frames,1:127,1:2) dimensions
        % processed_tap is (1:254)
        % diss is (1:1)

            tap_disps = tap_data - self.still_tap;

            processed_tap = [tap_disps(:,:,1) tap_disps(:,:,2)];

            % dissimilarity measure 
            differences = self.ref_tap - tap_disps;
            diss = norm([differences(:,:,1)'; differences(:,:,2)']);

        end%good
        
%         function [dissims, y_processed] = process_taps(self,radii_data)
%         % Return modified y data (1 by 256), dissimilarity data and the minimum x point
%         % for a single radius of data (intended to be a single radius).
% 
%             y_processed = [];
%             dissims = [];
% 
%             for tap_num = 1:length(radii_data) %TODO? is length safe, maybe use size(,)?
%                 current_tap_data_norm = radii_data{1,tap_num}(: ,:  ,:)- radii_data{1,tap_num}(1 ,:  ,:);
%                 
%                 n_pins = size(radii_data{1,1},2);
%                 max_i = zeros(1, n_pins);
%                 for pin = 1:n_pins
% 
%                     [~,max_ind_x]=max(abs(current_tap_data_norm(: ,pin  ,1)));
%                     [~,max_ind_y]=max(abs(current_tap_data_norm(: ,pin  ,2)));
% 
%                     max_i(1,pin) = max_ind_x;
%                     max_i(2,pin) = max_ind_y;
%                 end
% 
%                 average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2)); % want to compare same frame across tap, not different frames for each pin
% 
%                 differences = self.ref_diffs_norm(self.ref_diffs_norm_max_ind ,:  ,:) ...
%                               - current_tap_data_norm(average_max_i,:,:); 
% 
%                 y_processed = [y_processed;...
%                               current_tap_data_norm(average_max_i,:,1) current_tap_data_norm(average_max_i,:,2)]; %#ok<AGROW>
% 
%                 diss = norm([differences(:,:,1)'; differences(:,:,2)']);
%                 dissims =[dissims diss]; %#ok<AGROW>
%             end
%         end%good

        function [dissims, y_processed] = process_taps(self,radii_data)
        % Return modified y data (1 by 256), dissimilarity data and the minimum x point
        % for a single radius of data (intended to be a single radius).

            y_processed = [];
            dissims = [];

            for tap_num = 1:length(radii_data) %TODO? is length safe, maybe use size(,)?
                tap_disps = radii_data{1,tap_num} - self.still_tap;

                differences = self.ref_tap - tap_disps; 

                y_processed = [y_processed;...
                              tap_disps(:,:,1) tap_disps(:,:,2)]; %#ok<AGROW>

                diss = norm([differences(:,:,1)'; differences(:,:,2)']);
                dissims =[dissims diss]; %#ok<AGROW>
            end
        end%good
        
    end%methods
    
end%class