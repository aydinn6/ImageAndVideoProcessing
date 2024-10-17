% Convert RGB to YUV and apply chroma subsampling

% Load an RGB image
rgb_image = imread('.png');

% Normalize (to range 0-1)
rgb = double(rgb_image) / 255;

% Define the YUV transformation matrix
transform_matrix = [0.299, 0.587, 0.114;
                    -0.14713, -0.28886, 0.436;
                    0.615, -0.51499, -0.10001];

% Convert to YUV
[M, N, ~] = size(rgb);
yuv = zeros(M, N, 3);  % Create output matrix
for i = 1:M
    for j = 1:N
        rgb_pixel = squeeze(rgb(i, j, :));  % Get pixel RGB values
        yuv(i, j, :) = transform_matrix * rgb_pixel;  % Apply transformation
    end
end

% Chroma subsampling (for U and V channels)
yuv_subsampled = zeros(M, N, 3);
yuv_subsampled(:,:,1) = yuv(:,:,1);  % Keep Y channel as is

% Subsample U and V channels
for i = 1:2:M  % Take every second row
    for j = 1:2:N  % Take every second column
        yuv_subsampled(i, j, 2) = yuv(i, j, 2);  % U channel
        yuv_subsampled(i, j, 3) = yuv(i, j, 3);  % V channel
    end
end

% Scale U and V channels to range 0-255
yuv_subsampled(:,:,2:3) = round(yuv_subsampled(:,:,2:3) * 255);
yuv_subsampled(:,:,1) = round(yuv_subsampled(:,:,1) * 255);

% Display results
figure;
subplot(1, 2, 1);
imshow(rgb_image);
title('RGB Image');

subplot(1, 2, 2);
imshow(uint8(yuv_subsampled));  % Show YUV image
title('YUV Image (Subsampled)');
