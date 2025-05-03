clc;
clear all;
close all;

% - Section 1 - %
A = [1 2 4 5;
     4 5 1 2;
     1 3 2 5;
     5 3 1 2];
B = [1 2 3 4];
C = [2;5;4;1];

disp('A ='); disp(A)
disp('B ='); disp(B)
disp('C ='); disp(C)

% - Section 2 - %
dot_product_of_BC = dot(B,C);
disp('Dot product of B and C ='); disp(dot_product_of_BC) 

% - Section 3 - %
CB = C * B;         
CAB = CB * A;      
CBAB = CAB * B'; 
disp('(CB)AB:'); disp(CBAB)

% - Section 4 - %
[U, S, V] = svd(A);
disp('U:'); disp(U)
disp('S:'); disp(S)
disp('V:'); disp(V)

% - Section 5 - %
eigenvalues = eig(A);
disp('Eigenvalues of A matrix:');
disp(eigenvalues);

% - Section 6 - %
rgb_img = imread('image.jpg');
figure; % Create a new figure window for RGB image
imshow(rgb_img); 
title('RGB Image');

% - Section 7 - %
% Red Channel
R_channel = rgb_img;
R_channel(:, :, 2) = 0; 
R_channel(:, :, 3) = 0; 

% Green Channel
G_channel = rgb_img;
G_channel(:, :, 1) = 0; 
G_channel(:, :, 3) = 0; 

% Blue Channel
B_channel = rgb_img;
B_channel(:, :, 1) = 0; 
B_channel(:, :, 2) = 0; 

% Display each channel in separate figures
figure; % New figure window for Red Channel
imshow(R_channel);
title('Red Channel');

figure; % New figure window for Green Channel
imshow(G_channel);
title('Green Channel');

figure; % New figure window for Blue Channel
imshow(B_channel);
title('Blue Channel');

% - Section 8 - %
gray_img = rgb2gray(rgb_img);
figure; % New figure window for Grayscale Image
imshow(gray_img);
title('Grayscale Image');

% - Section 9 - %
hsv_img = rgb2hsv(rgb_img);
figure; % New figure window for HSV Image
imshow(hsv_img);
title('HSV Image');

H_channel = hsv_img(:, :, 1); 
S_channel = hsv_img(:, :, 2); 
V_channel = hsv_img(:, :, 3); 

% Display each HSV channel in separate figures
figure; % New figure window for Hue Channel
imshow(H_channel);
title('Hue Channel');

figure; % New figure window for Saturation Channel
imshow(S_channel);
title('Saturation Channel');

figure; % New figure window for Value Channel
imshow(V_channel);
title('Value Channel');

% - Section 10 - %
[U, S, V] = svd(double(gray_img));

% - Compression with the first %80 parameters -
num_singular_values_80 = round(0.80 * min(size(S)));  % Number of singular values for the first 80%
S_80 = S(1:num_singular_values_80, 1:num_singular_values_80);
U_80 = U(:, 1:num_singular_values_80);
V_80 = V(:, 1:num_singular_values_80);

compressed_img_80 = uint8(U_80 * S_80 * V_80');  % Compressed image with the first 80% parameters

% - Compression with the first %40 parameters -
num_singular_values_40 = round(0.40 * min(size(S)));  % Number of singular values for the first 40%
S_40 = S(1:num_singular_values_40, 1:num_singular_values_40);
U_40 = U(:, 1:num_singular_values_40);
V_40 = V(:, 1:num_singular_values_40);

compressed_img_40 = uint8(U_40 * S_40 * V_40');  % Compressed image with the first 40% parameters

% - Display the original image -
figure;
imshow(gray_img);
title('Original Grayscale Image');

% - Display the compressed image with %80 parameters -
figure;
imshow(compressed_img_80);
title('Compressed (%80) Grayscale Image');

% - Display the compressed image with %40 parameters -
figure;
imshow(compressed_img_40);
title('Compressed (%40) Grayscale Image');

[U, S, V] = svd(double(gray_img));

% - Compression with the last %80 parameters -
num_singular_values_80_end = round(0.80 * min(size(S)));  % Number of singular values for the last 80%
S_80_end = S(end - num_singular_values_80_end + 1:end, end - num_singular_values_80_end + 1:end);  % Last %80
U_80_end = U(:, end - num_singular_values_80_end + 1:end);
V_80_end = V(:, end - num_singular_values_80_end + 1:end);

compressed_img_80_end = uint8(U_80_end * S_80_end * V_80_end');  % Compressed image with the last 80% parameters

% - Compression with the last %40 parameters -
num_singular_values_40_end = round(0.40 * min(size(S)));  % Number of singular values for the last 40%
S_40_end = S(end - num_singular_values_40_end + 1:end, end - num_singular_values_40_end + 1:end);  % Last %40
U_40_end = U(:, end - num_singular_values_40_end + 1:end);
V_40_end = V(:, end - num_singular_values_40_end + 1:end);

compressed_img_40_end = uint8(U_40_end * S_40_end * V_40_end');  % Compressed image with the last 40% parameters

% - Display the compressed image with the last %80 parameters -
figure;
imshow(compressed_img_80_end);
title('Compressed (Last %80) Grayscale Image');

% - Display the compressed image with the last %40 parameters -
figure;
imshow(compressed_img_40_end);
title('Compressed (Last %40) Grayscale Image');
