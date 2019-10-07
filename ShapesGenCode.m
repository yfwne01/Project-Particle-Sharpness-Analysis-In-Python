% Generating Shapes for Testing
% 04 - SEP - 2015, KS, Lee

clear
clc

%% Input Parameters

N = 6; % Total number of points
R = 0.25; % Inner radius

% For given number of vertices and ratio of inscribed & circumscried
% circles, we can generate random shapes by using one or both of the
% following methods:
% a) varying the angles subtended by two consecutive points
% b) varying the location of the points between inner and outer circles

Aspread = 'reg'; % Options: reg, ran
Rspread = 'alt'; % Options: reg, alt, ran

%% Shape Generation

Th = zeros(N, 1);

if (strcmp(Aspread, 'reg'))

    Th = 2 * pi / N * ones(N, 1);
    
else
    
    S = 2 * pi;
    
    for i = 1:N - 1
        
        Th(i) = S / (N - i + 1) * 2 * rand;
        
        S = S - Th(i);
        
    end
    
    Th(N) = S;
    
end

if (strcmp(Rspread, 'reg'))

    PS = num2str(dec2bin(2^N-1));
    
elseif (strcmp(Rspread, 'alt'))
    
    if mod(N,2)
    
        fprintf('N should be even for alt shapes \n \n');
        return
    
    end
    
    PS = '';
    
    for i = 1:(N/2)
        
        PS = [PS '10'];
        
    end
    
else
    
    % Here, we'll generate a random sequence of 1's and 0's of length (N)
    % If the ith digit is 0, then ith point lies on the inner circle
    % If the ith digit is 1, then ith point lies on the outer circle
    PS = num2str(dec2bin(randi([2^(N - 1) 2^N-1])));

end

x = zeros(N + 1, 1);
y = zeros(N + 1, 1);

x(1) = 1;
y(1) = 0;

for i = 2:N
    
    if (str2double(PS(i)) == str2double(PS(i - 1)))
        
        r = 1;
        
    elseif (str2double(PS(i)) > str2double(PS(i - 1)))
        
        r = 1 / R;
        
    else
        
        r = R;
        
    end
    
    x(i) = (x(i - 1) * cos(Th(i)) - y(i - 1) * sin(Th(i))) * r;
    y(i) = (x(i - 1) * sin(Th(i)) + y(i - 1) * cos(Th(i))) * r;
    
end

% Closing the loop

x(N + 1) = x(1);
y(N + 1) = y(1);

% Centering the figure

x = x - mean(x);
y = y - mean(y);

% Rotating it by 90% for vertical bilateral symmetry when available
xt = x;
x = -y;
y = xt;

%% Plotting

fill(x,y,'k')
axis equal
axis(2 * [-1 1 -1 1])
axis off

%% Computing Angles

A = zeros(N, 1);

for j = 1:N
    
    i = j - 1;
    k = j + 1;
    
    if (i < 1)
        
        i = N;
        
    end
        
    T1 = (x(k) - x(j)) * (x(i) - x(j)) + (y(k) - y(j)) * (y(i) - y(j));
    T2 = sqrt((x(k) - x(j))^2 + (y(k) - y(j))^2);
    T3 = sqrt((x(i) - x(j))^2 + (y(i) - y(j))^2);
    
    A(j) = round((acos(T1 / (T2 * T3)) / pi * 180));
        
end