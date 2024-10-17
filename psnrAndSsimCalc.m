% Load the original image
originalImage = imread(".png");

% Convert the image to grayscale
originalImage = im2double(rgb2gray(originalImage));

% Add noise
noiseLevel = 0.1; % Noise level (between 0 and 1)
noise = noiseLevel * randn(size(originalImage)); % Gaussian noise
distortedImage = originalImage + noise; % Add noise to the original image

% Normalize the distorted image to keep values between 0 and 1
distortedImage = max(0, min(1, distortedImage)); % Clamp values

% Calculate PSNR
mse = mean((originalImage(:) - distortedImage(:)).^2);
if mse == 0
    psnrValue = Inf; % If images are identical, PSNR is infinite
else
    psnrValue = 10 * log10(1 / mse); % PSNR calculation formula
end

% Calculate SSIM
[ssimValue, ~] = ssim(originalImage, distortedImage);

% Display the results
fprintf('PSNR: %.2f dB\n', psnrValue);
fprintf('SSIM: %.4f\n', ssimValue);

% Show the images
figure;
subplot(1, 2, 1);
imshow(originalImage);
title('Original Image');

subplot(1, 2, 2);
imshow(distortedImage);
title('Image with Added Noise');
