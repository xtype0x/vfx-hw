% from https://www.youtube.com/watch?v=hjc0t31VF5I

clear all; clc;

% Declare variables
sigma = 1;
radius = 1;
threshold = 30000;
size = 2 * radius + 1;          % the size of matrix filter
dx = [-1 0 1; -1 0 1; -1 0 1];  % x derivative matrix
dy = dx';                       % y derivative matrix
g = fspecial('gaussian', max(1, fix(6*sigma)), sigma); % also try Gaussian, fspecial for each, for example Edge Detection


% Read image
source = imread('.\Test Data\1_parrington\pano.jpg');

% Get all reds
im = source(:,:,1);

% Step 1: Compute derivatives
Ix = conv2(im, dx, 'same'); % also try gradient
Iy = conv2(im, dy, 'same');

% Step 2: Compute
Ix2 = conv2(Ix.^2, g, 'same');
Iy2 = conv2(Iy.^2, g, 'same');
Ixy = conv2(Ix.*Iy, g, 'same');

% Step 3: Harris Corner
harris = (Ix2.*Iy2 - Ixy.^2)./(Ix2+Iy2 + eps);

% Step 4: FInd local maxima
mx = ordfilt2(harris, size.^2, ones(size)); % search this function

harris = (harris == mx) & (harris > threshold);

% plot
[rows, cols] = find(harris);
figure, image(source), axis image, colormap(gray), hold on,
plot(cols, rows, 'ys'), title('corners detected');
