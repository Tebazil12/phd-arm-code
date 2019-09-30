% collects training data over locations xs and objects cs
%
% N Lepora April 2018

killPython; close all; clear all; clear classes; clear figures; clc; % dbstop if error

% on/offline
ONLINE = true;

% video output
VIDEO_ON = true;
if VIDEO_ON; videos = [ "Data" ]; end % "Voronoi" "Offline"

% paths
if strfind(system_dependent('getos'), 'Linux') == 1
    dirPath = '/home/lizzie/git/tactile-core/matlab/experiments/TacTip-demos/exploration/data';

elseif strfind(system_dependent('getos'), 'Microsoft Windows') == 1
    dirPath = 'C:\Users\lizzie\Documents\Repos\tactile-core\matlab\experiments\TacTip-demos\exploration\data';

else
    ME = MException('MATLAB:UnknownOperatingSystem', ...
        'Unknown operating system: only windows and linux are supported');
    throw(ME) 
end


if ~ONLINE; dirTrain = 'trainCircleTap08021751'; end

%% load/create training

% set experiment training parameters
switch ONLINE
    case false
        load(fullfile(dirPath,dirTrain,'expt'), 'Expt')
        Expt.nSets = 1;
    case true
        Expt.actionTraj = [0 0 5 0 0 0; 0 0 0 0 0 0]; % tap move trajectory wrt tool/sensor frame
        Expt.robotSpeed = [25 15 15 10];%2*[50 30 15 10];
        Expt.workFrame = [273 -272 69 180 0 180]; % work frame wrt base frame (x,y,z,r,p,y), center if circle %find using abb jogger
        
        Expt.nSets = 1; % how many times to repeat the whole experiment
        Expt.sets = cellstr(num2str((1:Expt.nSets)','%02i'))';
        
        radius = 54;
        
        % tactip calibration
        Expt.sensorParams = [119.18 232.37 54.66 129.47 0.46 0.47 0.28]; % min_threshold max_threshold min_area max_area min_circularity min_convexity min_inertia_ratio
end
if VIDEO_ON;Expt.videos = videos; Expt.resolution = 400;end

% startup robot (real/dummy)
fileName = mfilename; 
commonStartupTrain

% startup model
tactile = TactileData(Expt);
%voronoi = VoronoiTactile(Expt);

%% collect training data
tap_index = 1;
% loop training runs
for iSet = 1:Expt.nSets       
        
    % loop angles
    for theta = -45:5:45
        fprintf('New Angle: %i\n', theta)
        
        % next position (above object)
        robot.move([0 0 0 0 0 theta])
        
        % initialize videos
        fileData = ['c' num2str(theta,'%02i') '_' num2str(iSet,'%02i')];
        if VIDEO_ON; for v = videos; video.initialize({[fileData '_' v{1}]},dirPath,dirTrain); end; end
        
        % loop through positions
        for displacement = -10:1:10
            fprintf('disp: %i\n', displacement)
            
            x = (radius+displacement) * sind(-theta);
            y = (radius+displacement) * cosd(-theta);
            new_position = [x y 0 0 0 theta];
            
            % start camera recording
            camera.start
            
            % next position
            robot.move(new_position)
            
            % collect data & store
            tacData = robot.recordAction;
            data{tap_index} = tacData;
            tap_index = tap_index +1;
            % stop camera recording
            camera.stop
            
            % display and write videos
            if VIDEO_ON
                for v = videos
                    evalc(['img = ' video.display(v) '(tacData);']);
                    video.write(v,img);
                end
            end
        end
        
        % save store per angle 
        if ONLINE; save(fullfile(dirPath,dirTrain,fileData), 'data'); end

        % close videos
        if VIDEO_ON
            for v = videos; video.close(v); end
        end
    end
    
    % untwist sensor
    for index = [1]
        robot.move([0 0 0 0 0 -45*index]) % rotate back to start pose: (190/2)=95 (as rotation by >=180 is dangerous)
    end
end

% exit
if ONLINE; save(fullfile(dirPath,dirTrain,'expt'), 'Expt'); end
commonShutdown
