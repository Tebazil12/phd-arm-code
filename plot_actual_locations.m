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
load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-14_1443/all_data.mat') %circle step20

% figure(2)
close all
the_figure = figure('position', [0, 0, 500, 500],'DefaultAxesFontSize',16);
hold on
set(0,'defaultAxesFontName', 'arial')
set(0,'defaultTextFontName', 'arial')

r = 53;

% --- https://uk.mathworks.com/matlabcentral/answers/3058-plotting-circles 
ang=0:0.01:2*pi; 
x=-53+r*cos(ang);
y=r*sin(ang);
plot(x,y,'color',[1 0.6 0],'LineWidth',2);
%---%

distances = sqrt((ex.dissim_locations(:,1)--53).^2 + ex.dissim_locations(:,2).^2) - r
abs_mean_distance = round(mean(abs(distances)), 2)
var_distnace = round(var(distances),2)
std_distnace = round(std(distances),2)

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

xlabel("x displacement (mm)",'FontSize',16)
ylabel("y displacement (mm)",'FontSize',16)
% title({"Square"},'FontSize',11)

% legend([pg pf pd],{'Tap location','Robot Motion','Predicted Edge'},'FontSize',8,'location','best')

% %for flower
% axis([-135 25 -80 80])
% xticks(-120:20:20)

% %for circle
axis([-125 15 -70 70])
xticks(-120:20:20)

% %for banana
% axis([-55 45 -70 65]) % 100 x 135
% xticks(-120:20:60)

% %for brick
% axis([-80 15 -60 60]) %95 x 120
% xticks(-120:20:20)

% grid on
% plot([0 0],[-80 80],'Color',[0.7 0.7 0.7])
% plot([-120 40],[0 0],'Color',[0.7 0.7 0.7])
% axis equal
daspect([1 1 1])

% axis tight; 

% BACKGROUND IMAGES
% I = imread('pics/banana3-2.png'); 
% I = imread('pics/brick3.png'); 
% h = image(xlim,ylim,flipdim(I, 1)); 
% uistack(h,'bottom')


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

