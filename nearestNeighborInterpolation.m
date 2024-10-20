clc, close all, clear all;
% Load the image
img = imread('.png'); % Enter the name of your image file here

% Scaling factors
scale_factor = 3; % Scale the image by 200%

% Get original dimensions
[original_height, original_width, channels] = size(img);

% Calculate new dimensions
new_height = round(original_height * scale_factor);
new_width = round(original_width * scale_factor);

% Create new image
scaled_img = zeros(new_height, new_width, channels, 'uint8');

% Nearest neighbor interpolation for scaling
for i = 1:new_height
    for j = 1:new_width
        % Find the corresponding pixel position in the original image
        orig_i = round(i / scale_factor);
        orig_j = round(j / scale_factor);
        
        % Boundary check
        orig_i = min(max(orig_i, 1), original_height);
        orig_j = min(max(orig_j, 1), original_width);
        
        % Assign the pixel value to the new image
        scaled_img(i, j, :) = img(orig_i, orig_j, :);
    end
end

% Visualize the results
figure; imshow(img);
title('Original Image');

figure; imshow(scaled_img);
title('Scaled Image with Nearest Neighbour Interpolation');
