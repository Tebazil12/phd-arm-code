figure(3)
clf
hold on

x1 = 1;
y1=1;

l = -sqrt(8);

%%%THETA IS THE ANGLE YOU WANT THE TIP AT
theta = 10

% for moving radially line (green)

x2 = x1 + l*cosd(theta);
y2 = y1 + l*sind(theta);


plot([x1 x2],[y1 y2],'g')
          




axis equal



l = sqrt(8)
%for tapping tangentially/extrapolating (red)
x2 = x1 + l*sind((-theta));
y2 = y1 + l*cosd((-theta));
plot([x1 x2],[y1 y2],'r')
plot([1 3],[1 -1],'k:')
plot([1 -1],[1 -1],'k:')
plot([1 -1],[1 3],'k:')
plot([1 3],[1 3],'k:')
plot([1 1],[-1 3],'k')
plot([-1 3],[1 1],'k')
axis equal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NEW THETA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% to get angle between last two points 
theta2 = -atan2d((x22-x1),(y22-y1))

% radially (blue)
x2 = x1 + l*cosd(theta2);
y2 = y1 + l*sind(theta2);

plot([x1 x2],[y1 y2],'b')

theta
theta2

%tangentially (yellow)
x2 = x1 + l*sind((-theta2));
y2 = y1 + l*cosd((-theta2));
plot([x1 x2],[y1 y2],'y')


% green == -blue
% yellow == red



% mod(atan2d((y2-y1),(x2-x1))+180,180)
% -(atan2d((y2-y1),(x2-x1)) +180)
% -(atan2d((y2-y1),(x2-x1)) -180)
p1=    [-3.04438223136478,0.590987102066980];
p2=    [-1.59976169440578,5.51853923836783];
p3=    [-9.21112609898898,-2.67091324031742];
p4=    [-2.48517653298946,-2.77862784299948];

theta3 =  atan2d((p2(2)-p1(2)),(p2(1)-p1(1)))

ang = -29.5

