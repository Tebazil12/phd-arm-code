% Copyright (c) 2019 Elizabeth A. Stone
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.


%% Dissim found by:
function [processed_tap, diss] = process_single_tap(self,tap_data)
% Tap data should be (1:n_frames,1:n_pins,1:2) dimensions
% processed_tap is (1:(2*n_pins))
% diss is (1:1)

    % Remove "non contact" frame from all taps to get just pin displacements
    current_tap_disps = tap_data(: ,:  ,:) - tap_data(1 ,:  ,:);
    
    % Find how many pins this data actually has
    n_pins = size(tap_data,2);
    
    max_i = zeros(1, n_pins); %init empty
    
    for pin = 1:n_pins
        
        % Find which frame has max disp for each pin in each dimension (x,y)
        [~,max_ind_x]=max(abs(current_tap_disps(: ,pin  ,1)));
        [~,max_ind_y]=max(abs(current_tap_disps(: ,pin  ,2)));

        max_i(1,pin) = max_ind_x;
        max_i(2,pin) = max_ind_y;
    end

    % Find max frame on avarage (NB want to compare same frame across tap, not
    % different frames for each pin)
    average_max_i = round(mean([max_i(1,:)  max_i(2,:)],2)); 

    % Return tap at max displacement
    processed_tap = [current_tap_disps(average_max_i,:,1) current_tap_disps(average_max_i,:,2)];

    % difference between new tap and reference tap
    differences = self.ref_tap_disps(self.ref_tap_disps_max_ind,:,:) - current_tap_disps(average_max_i,:,:); 
    
    % Calc dissimilarity measure 
    diss = norm([differences(:,:,1)'; differences(:,:,2)']);
end


%% Reference tap found by:

% Get displacement not just absolute position
ex.ref_tap_disps = ref_tap(: ,:  ,:) - ref_tap(1 ,:  ,:); % assumes starts on no contact/all start in same position

% find the frame in ref_tap_disps with greatest displacement
[~,an_index] = max(abs(ex.ref_tap_disps));
ex.ref_tap_disps_max_ind = round(mean([an_index(:,:,1) an_index(:,:,2)]));