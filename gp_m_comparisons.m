%% Generate some random data
clearvars
clf

x_matrix = (0:0.05:1) + randn(1,21)*0.001;
x_matrix = x_matrix';
% x_matrix= [0.104876208155542;0.182617639638471;0.250174468413053;0.333769721387247;0.383203408868252;0.553668767772118;0.606996163929327;0.795384421908286;1.05650352742386;0.970373029144668];
% x_matrix = [0.159647362149247;0.108863063738743;0.132150723282020;0.321628483089092;0.649394676863718;0.652154524665157;0.591260336263522;0.933271723609033;0.815431646025134;0.996342818229817];
x_matrix = [0.000869528138096387;0.0497937504978302;0.101147236389465;0.150083074768201;0.200472195783510;0.250429324255031;0.300302125397144;0.350013141540285;0.399989025802214;0.449276663447137;0.501178387289419;0.549701937456157;0.599958217515006;0.648926617205165;0.700031311687550;0.751500224264257;0.798887467208279;0.850721176505681;0.898535293209997;0.950536052303514;1.00153804876904];

y = sin(2*pi .* x_matrix) + randn(21,1)*0.2;
% y = [0.806620067646877;1.00930145769205;1.27009857457890;0.863355678856253;0.649853059775330;-0.361113723861702;-0.866262866147843;-0.553117667295003;0.0622135239397272;-0.209876183076949];
% y = [0.702783618313878;0.629320688247004;0.601044283720058;0.751954389267647;-0.955289576581552;-0.410031842867234;-0.943676508245586;-0.202468819169958;-1.00932731392046;0.0872537100360546];
y =[0.0302786649353048;0.0767636017959244;0.937376639963559;0.599316971822375;0.817374995291669;1.08761844186300;1.02547969081149;0.793620271004800;0.638459012960436;0.400820751484652;-0.0411787403801558;-0.254096327360869;-0.661672533729863;-0.909068524860618;-1.05330789263268;-0.984165935022226;-1.15519307062026;-0.946050980280817;-0.619306461668225;-0.234286813905780;-0.0312462545635758];

hold on
% plot this random data
scatter(x_matrix,y,'+');

xlabel('x')
ylabel('y')

%% Define hyper parameters - for testing only
sigma_f = 0.9;
l = 0.1;
sigma_n = 0.4;

%% Optimize hyper-params
% [par, fval, flag] = fminsearch(@(mypar)to_max(mypar(1), mypar(2), mypar(3), y , x_matrix), [0.5 0.2 0.2] )
[par, fval, flag] = fminunc(@(mypar)to_max(mypar(1), mypar(2), mypar(3), y , x_matrix), [0.5 0.2 0.2] )
 
if flag ~= 1
    warning("fminsearch was not happy")
    flag
end 

if round(par(1),1) == 0 || round(par(2),1) == 0 || round(par(3),1) == 0
    warning("A hyper-parameter is zero! Bad fit")
end

sigma_f = par(1);
l = par(2);
sigma_n = par(3);

%% Validate over many points
j = 0;
for l = [par(2) 0.12:0.02:0.4]  
    k_cap = calc_k_cap(x_matrix, sigma_f,l, sigma_n);

    i = 1;
    for x_star = 0:0.01:1.2
        x_stars(i) = x_star;

        % setup covariance matrix stuff
        k_star      = calc_k_star(x_star, x_matrix, sigma_f,l, sigma_n);
        k_star_star = calc_covar_ij(x_star, x_star, sigma_f,l, sigma_n);

        % Estimate y
        y_star(i) = k_star * inv(k_cap) * y;

        % Estimate variance
        var_y_star(i) = k_star_star - (k_star * inv(k_cap) * transpose(k_star));
        if var_y_star(i) < 0.0000
            var_y_star(i) =0; % otherwise -0.0000 causes errors with sqrt()
        end

        i = i+1;
    end

    % plot Stdev band
    figure(1)
    subplot(4,4,j+1)
    hold on
    if j == 0
        str = sprintf('Optimum l: %g,\n \\sigma_n: %g, \\sigma_f: %g', l, sigma_n,sigma_f);
        title(str)
    else
        str = sprintf("l: %g", l);
        title(str) 
    end
    
    fill([x_stars, fliplr(x_stars)],...
         [y_star+sqrt(var_y_star), fliplr(y_star-sqrt(var_y_star))],...
         [1 1 0.8],...
         'EdgeColor','none')
    plot(x_stars, y_star+sqrt(var_y_star), 'Color',[ 1 0.8 0.2] )
    plot(x_stars, y_star-sqrt(var_y_star), 'Color',[ 1 0.8 0.2] )

    % plot predictions from gp
    plot(x_stars,y_star, 'k') 

    % plot actual input data
    scatter(x_matrix,y,'+','b');

    % plot actual sine wave (no noise)
    plot((0:0.01:1), sin(2*pi .* (0:0.01:1)), 'r')

    hold off
    j = j +1;
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--------------------------- FUNCTIONS ---------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 function val = to_max(sigma_f, l, sigma_n, y, x_matrix)
%     sigma_f
%     l
%     sigma_n
    k_cap = calc_k_cap(x_matrix, sigma_f, l, sigma_n);
    
    part1 = -0.5 * y' * pinv(k_cap) * y;
    part2 = -0.5 * log(det(k_cap)); % NB, |K| means det(K), not abs(K) or norm(K)
    part3 = -0.5 * length(y)*log(2*pi);
    neg_val = part1 + part2 + part3;
    val = - neg_val; % because trying to find max with a min search

%     [n, ~] = size(k_cap);
%     R = chol(k_cap); % K = R'*R
%     y';
%     % Calculate the log determinant of the covariance matrix
%     logdet_K = sum(2*log(R(1:n+1:end)));
% 
% 
%     % Calculate the log likelihood
%     alpha = R \ (R' \ y);
% %     val = 0.5*mean(sum(y'.*alpha, 1)) + 0.5*logdet_K + 0.5*n*log(2*pi);
%     val = (0.5* y' * (R' * R)\y )+ 0.5*logdet_K + 0.5*n*log(2*pi);

 end
 
% Define covariance function, k:
function k = calc_covar_ij(x,x_prime,sigma_f,l, sigma_n)
    if x == x_prime
        kron_delta_func = 1;
    else
        kron_delta_func = 0;
    end
    k = (sigma_f^2 * exp( (-(x - x_prime)^2)/ (2*(l^2)) )) + ((sigma_n^2)*kron_delta_func);
end

function k_cap = calc_k_cap(x_matrix, sigma_f,l, sigma_n)
    size_x = length(x_matrix);
    k_cap = zeros(size_x,size_x);
    for i = 1:size_x 
        for j = 1:size_x
            k_cap(i,j) = calc_covar_ij(x_matrix(i),x_matrix(j), sigma_f,l, sigma_n);
            
        end
    end
    k_cap;
end

function k_star = calc_k_star(x_star, x_matrix, sigma_f,l, sigma_n)
    size_x = length(x_matrix);
    k_star = zeros(1,size_x);
    for i = 1:size_x
        k_star(i) = calc_covar_ij(x_star, x_matrix(i), sigma_f,l, sigma_n);
    end
end



