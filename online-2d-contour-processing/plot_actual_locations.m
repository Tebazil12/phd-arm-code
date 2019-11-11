% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1547/all_data.mat') %banana
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1515/all_data.mat') %brick
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1156/all_data.mat') %square


% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1442/all_data.mat') %flower pt6
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1436/all_data.mat') %circle pt6
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1426/all_data.mat') %flower pt11
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1419/all_data.mat') %circle pt11
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1406/all_data.mat') %flower pt21
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1357/all_data.mat') %circle pt21

% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1406/all_data.mat') %flower step5
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-14_1116/all_data.mat') %flower step10
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1357/all_data.mat') %circle step5
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-14_1110/all_data.mat') %circle step10
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-14_1427/all_data.mat') %circle step15
% load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-14_1443/all_data.mat') %circle step20

% higher_path = '/home/lizzie/OneDrive/data/';
% file_name = '/all_data.mat';

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_1950'; shape = "circle";%NODISS circle
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2002'; shape = "circle";%NODISS circle
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2029'; shape = "banana";%NODISS banana
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2039'; shape = "flower"; %NODISS flower
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-02_2051'; shape = "flower"; %NODISS flower 6pt

% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1721'; shape = "flower"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1728'; shape = "flower"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1737'; shape = "flower"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1750'; shape = "flower"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1744'; shape = "flower"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1759'; shape = "circle"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1817'; shape = "circle"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1805'; shape = "circle"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1824'; shape = "circle"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1851'; shape = "banana"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1931'; shape = "banana"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1954'; shape = "brick"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2021'; shape = "brick"; %NODISS circle 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2006'; shape = "brick"; %NODISS circle 6pt

% failure tests
% file_name = '/matlab.mat';
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2013'; shape = "brick"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1837'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1830'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1857'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1845'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1908'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1905'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1915'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1923'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1927'; shape = "banana"; %NODISS flower 6pt
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_1940'; shape = "banana"; %NODISS flower 6pt ...
% current_folder = 'runrobot_2d_contouring_NO_DISSIM2019-11-03_2006'; shape = "brick"; %NODISS flower 6pt




% full_path = strcat(higher_path, current_folder, file_name);
% load(full_path)

% figure(2)
% close all
% the_figure = figure('position', [0, 0, 500, 500],'DefaultAxesFontSize',16);
% hold on
% set(0,'defaultAxesFontName', 'arial')
% set(0,'defaultTextFontName', 'arial')
hold on
if shape == "flower"
    %for flower
    axis([-135 25 -80 80])
    xticks(-120:20:20)
    
elseif shape == "circle"

    % %for circle
    axis([-125 15 -70 70])
    xticks(-120:20:20)
    
elseif shape == "banana"    

    %for banana
    axis([-55 45 -70 65]) % 100 x 135
    xticks(-120:20:60)
    
    % BACKGROUND IMAGES
    I = imread('/home/lizzie/OneDrive/pics/banana3-2.png'); 
    h = image(xlim,ylim,flipdim(I, 1)); 
    uistack(h,'bottom')

elseif shape == "brick"
    %for brick
    axis([-80 15 -60 60]) %95 x 120
    xticks(-120:20:20)
    
    % BACKGROUND IMAGES
    I = imread('/home/lizzie/OneDrive/pics/brick3.png'); 
    h = image(xlim,ylim,flipdim(I, 1)); 
    uistack(h,'bottom')
    
end

if shape == "circle"
    r = 53;

    % --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles 
    ang=0:0.01:2*pi; 
    x=-53+r*cos(ang);
    y=r*sin(ang);
    plot(x,y,'color',[1 0.6 0],'LineWidth',2);
    %---%
    r2=r+2;
    ang=0:0.01:2*pi; 
    x=-53+r2*cos(ang);
    y=r2*sin(ang);
    plot(x,y,'color',[0 1 0],'LineWidth',1);
    
    r3=r-2;
    ang=0:0.01:2*pi; 
    x=-53+r3*cos(ang);
    y=r3*sin(ang);
    plot(x,y,'color',[0 1 0],'LineWidth',1);

    distances = sqrt((ex.dissim_locations(:,1)--53).^2 + ex.dissim_locations(:,2).^2) - r
    abs_mean_distance = round(mean(abs(distances)), 2)
    var_distnace = round(var(distances),2)
    std_distnace = round(std(distances),2)
end

%square
% x = [-70 0 0 -70 -70]
% y = [-40 -40 40 40 -40]
% plot(x,y,'color',[1 0.6 0])

%brick
% w = 4;
% h = 1;
% x = [-50 0 0 -50 -50]-w;
% y = [-37.5 -37.5 37.5 37.5 -37.5]+h;
% plot(x,y,'color',[1 0.6 0])

xs = [];
ys=[];
for a= 1:size(ex.actual_locations,2)
    for b = 1:size(ex.actual_locations{a},2)
        xs = [xs; ex.actual_locations{a}{b}(1)];
        ys = [ys; ex.actual_locations{a}{b}(2)];
%         if mod(a,2) == 0 
%             pa = plot(ex.actual_locations{a}{b}(1),ex.actual_locations{a}{b}(2),'Color',[0 0 1]);
%             pa.Marker = '+';
%         elseif mod(a,2) == 1 
%             pb = plot(ex.actual_locations{a}{b}(1),ex.actual_locations{a}{b}(2),'Color',[0 0.7 1]);
%             pb.Marker = '+';
% %             pb.MarkerFaceColor= 'r';
% %         elseif mod(a,2) == 2 
% %             pc = plot(ex.actual_locations{a}{b}(1),ex.actual_locations{a}{b}(2),'g+');
%         end
%         pause(1)
    end
end




% 
% plot(ex.dissim_locations(:,1),ex.dissim_locations(:,2),'o','Color',[1 0.6 0])
% plot(ex.dissim_locations(:,1),ex.dissim_locations(:,2),'y','Color',[1 0.6 0])


pd = plot(ex.dissim_locations(:,1),ex.dissim_locations(:,2),'k-','LineWidth',2,'MarkerSize',10,'MarkerFaceColor','k');
% pe = plot(ex.dissim_locations(:,1),ex.dissim_locations(:,2),'k');


pf=plot(xs,ys,'Color',[0 0.7 1],'LineWidth',1);
pg=plot(xs,ys,'+','Color',[1 0 0],'LineWidth',1.25);

xlabel("x displacement (mm)")%,'FontSize',16)
ylabel("y displacement (mm)")%,'FontSize',16)
% title({"Square"},'FontSize',11)

% legend([pg pf pd],{'Tap location','Robot Motion','Predicted Edge'},'FontSize',8,'location','best')



% grid on
% plot([0 0],[-80 80],'Color',[0.7 0.7 0.7])
% plot([-120 40],[0 0],'Color',[0.7 0.7 0.7])
% axis equal
daspect([1 1 1])

% axis tight; 




% grid on
% grid minor 

%min side of objects:
%banaana [-50 40 -70 60]
% brick  [-70 10 -55 60]
% flower [-130 10 -70 70]
%
%all [-130 40 -85 85] - not equal!
%flower biggest: [-130 10 -70 70]
%circ biggest:
%odd items:

% box on
% 
% print('-dpng', ['/home/lizzie/OneDrive/matlab-figs/', current_folder,'.png']);
% savefig(['/home/lizzie/OneDrive/matlab-figs/', current_folder,'.fig']);
