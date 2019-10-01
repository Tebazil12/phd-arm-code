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
        Expt.workFrame = [273-54 -272 67 180 0 180];%[475 180 69 180 0 180]; % board 2 ABB1 % specify work frame wrt base frame (x,y,z,r,p,y) %find using abb jogger
        
        Expt.nSets = 1; % how many times to repeat the whole experiment
        Expt.sets = cellstr(num2str((1:Expt.nSets)','%02i'))';
        
        angles = [-160:20:180];
        Expt.nxs = 21; % num of displacements per tactip orientation
        Expt.ncs = size(angles,2);  % num of tactip orientations 
        Expt.xs = [-linspace(-10,10,Expt.nxs); zeros(5,Expt.nxs)]';
        Expt.cs = [zeros(5,Expt.ncs); -angles]';
        
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

% loop training runs
for iSet = 1:Expt.nSets
    
    % loop objects
    for ic = 1:Expt.ncs
        fprintf('set %i/%i object %i/%i\n', [iSet Expt.nSets ic Expt.ncs])
        
        % next position (above object)
        mach.moveAbsoluteWhat(ic)
        robot.move(mach.pose)
        
        % initialize videos
        fileData = ['c' num2str(ic,'%02i') '_' num2str(iSet,'%02i')];
        if VIDEO_ON; for v = videos; video.initialize({[fileData '_' v{1}]},dirPath,dirTrain); end; end
        
        % loop through positions
        for ix = 1:Expt.nxs
            fprintf('where %i/%i\n', [ix, Expt.nxs])
            
            % start camera recording
            camera.start
            
            % next position
            mach.moveAbsoluteWhere(ix)
            robot.move(mach.pose)
            
            % collect data & store
            tacData = robot.recordAction;
            data{ix} = tacData;

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
        
        % save store per object
        if ONLINE; save(fullfile(dirPath,dirTrain,fileData), 'data'); end

        % close videos
        if VIDEO_ON
            for v = videos; video.close(v); end
        end
    end
    
    % exit: return robot
    for ic = Expt.ncs:-1:1
        mach.moveAbsoluteWhat(ic)
        robot.move(mach.pose)
    end
end

% exit
if ONLINE; save(fullfile(dirPath,dirTrain,'expt'), 'Expt'); end
commonShutdown
