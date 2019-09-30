figure(6)
clf
for i = 1:4
    if i == 1
        load('/home/lizzie/git/masters-tactile/offline_singlets_1inputs.mat')
        colour = [ 0 .8 1];
        marker_t = 'o';
    elseif i==2
        load('/home/lizzie/git/masters-tactile/offline_singlets_2inputs.mat')
        colour = [.95 0 .95];
        marker_t = 's';
    elseif i==3
        load('/home/lizzie/git/masters-tactile/offline_singlets_3inputs.mat')
        colour = 'b';
        marker_t = 'd';
    elseif i==4
        load('/home/lizzie/git/masters-tactile/offline_singlets_5inputs.mat')
        colour = [0 .85 0];
        marker_t = 'p';
    end

figure(6)
% clf
hold on

% fill([-15 15 15 -15], [2 2 -2 -2], [1 1 0.8],'LineStyle','none','facealpha',.5)
% fill([-15 15 15 -15], [1 1 -1 -1], [0.8 1 0.8],'LineStyle','none','facealpha',.5)
% plot(x_real,x_real_test - x_real,'.')

error_x =  x_real - x_real_test;
% remove_x_anomolies = 

pl{i} = plot(x_real, error_x,marker_t,'Color',colour,'MarkerSize',3,'MarkerFaceColor',colour);

% stem(x_real,error_x,'ob','filled','MarkerSize',3.5)
% stem(x_real(:,1),error_x,'o','filled','MarkerSize',3.5)
% plot(x_real,error_x)
title("Error in Displacement Predictions")
ylabel("Error in predicted displacement  (mm)")
xlabel("Real displacement (mm)")
grid on

% axis equal
plot([-12 12],[0 0],'k')
axis([-10.5 11.5 -7 14])
% axis([-10.5 11.5 -4 4])
end
grid minor
legend([pl{1}(1) pl{2}(1) pl{3}(1) pl{4}(1)],{'1 Training Line','2 Training Lines','3 Training Lines','5 Training Lines'},'Location','northwest')
hold off

figure(7)
clf

for i = [1 2 3 5]
    if i == 1
        load('/home/lizzie/git/masters-tactile/offline_singlets_1inputs.mat')
        colour = [ 0 .8 1];
    elseif i==2
        load('/home/lizzie/git/masters-tactile/offline_singlets_2inputs.mat')
        colour = [.95 0 .95];
    elseif i==3
        load('/home/lizzie/git/masters-tactile/offline_singlets_3inputs.mat')
        colour = 'b';
    elseif i==5
        load('/home/lizzie/git/masters-tactile/offline_singlets_5inputs.mat')
        colour = [0 .85 0];
    end

figure(7)
% clf
hold on

% fill([-15 15 15 -15], [2 2 -2 -2], [1 1 0.8],'LineStyle','none','facealpha',.5)
% fill([-15 15 15 -15], [1 1 -1 -1], [0.8 1 0.8],'LineStyle','none','facealpha',.5)
% plot(x_real,x_real_test - x_real,'.')

error_x =  x_real - x_real_test;
((error_x > 14) + (error_x < -14)) .* error_x;

a =sum(sum(error_x > 14));
b= sum(sum(error_x < -14));
error_x > 14;
a+b
error_x(abs(error_x)>14) = NaN; % anomolies removed
error_x(abs(error_x)<-14) = NaN;

bar(i, nanmean(nanmean(abs(error_x))), 'FaceColor',colour );

% stem(x_real,error_x,'ob','filled','MarkerSize',3.5)
% stem(x_real(:,1),error_x,'o','filled','MarkerSize',3.5)
% plot(x_real,error_x)
title("Mean Error in Displacement Predictions")
ylabel("Mean error in predicted displacement  (mm)")
xlabel("Number of training lines")
grid on

axis([0.4 5.5 0 1.601])
xticks([1 2 3 5])


end
grid minor
set(gca,'XGrid','off','XMinorGrid','off')
% legend([pl{1}(1) pl{2}(1) pl{3}(1) pl{4}(1)],{'1 Input Line','2 Input Lines','3 Input Lines','5 Input Lines'},'Location','northwest')
hold off