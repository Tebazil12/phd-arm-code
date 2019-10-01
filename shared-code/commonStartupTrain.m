% script for common startup

% create train directory
if ONLINE
    dirTrain = [fileName datestr(now,'yyyy-mm-dd_HHMM')];
    mkdir(fullfile(dirPath,dirTrain)); 
end


if ~ONLINE; data = loadData(fullfile(dirPath,dirTrain), Expt); end

% create file of metadata 
info_file = fopen(fullfile(dirPath,dirTrain,'README'),'w');
fprintf(info_file, fullfile(dirPath,dirTrain));
fprintf(info_file, '\r\n');
[~,repo]=system('git config --get remote.origin.url');
fprintf(info_file,'\r\nCurrent git repo: %s' ,repo);
[~,current_head] = system('git rev-parse --short HEAD');
fprintf(info_file,'\r\nCurrent git HEAD: %s' ,current_head);
[~,branches] = system('git branch');
fprintf(info_file,'\r\nCurrent branch:\r\n %s', branches);
fprintf(info_file, '\r\nExperiment Description:\r\n');
fprintf(info_file, '-----------------------\r\n');
fprintf(info_file, '-45 to 45 deg, 5 deg spacing, -10 to 10mm, 1mm spacing, static radius for all.3d with depth -2:0.5:2\r\n');
fclose(info_file);

% startup machine and agent
%mach = Machine(Expt);

% startup robot
if ONLINE; robotArm = ABBRobotArm; end
if ~ONLINE; robotArm = DummyRobotArm; end
robotArm.setSpeed(Expt.robotSpeed)

% startup sensor
if ONLINE && ~isfield(Expt,'sensorParams'); sensor = TacTip; end
if ONLINE && isfield(Expt,'sensorParams') 
    par = Expt.sensorParams;
    sensor = TacTip('min_threshold',par(1),...
                    'max_threshold',par(2),...
                    'min_area',par(3),...
                    'max_area',par(4),...
                    'min_circularity',par(5),...
                    'min_convexity',par(6),...
                    'min_inertia_ratio',par(7));
end
if ~ONLINE; sensor = TactileTest(mach, data); end
robot = TactileActionRobot(robotArm, sensor, Expt.workFrame, Expt.actionTraj);

% detect and choose pins
rad = 300; mdist = 0;
if ONLINE; Expt.pinPositions = robot.initPinPositions(rad,mdist); end
if ~ONLINE; robot.setPinPositions(Expt.pinPositions); end
[Expt.nPins, Expt.nDims] = size(Expt.pinPositions);

% startup camera
camera = Camera(Expt);
camera.initialize(dirPath, dirTrain, [], [])

% startup video
video = Video(Expt);
