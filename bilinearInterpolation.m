clc, close all, clear all;
% The image to be scaled is read
im = imread(".png");
out_dims = [270, 396];
% First, the original image size and the desired image dimensions are assigned to fixed variables
in_rows = size(im,1);
in_cols = size(im,2);
out_rows = out_dims(1);
out_cols = out_dims(2);
% The ratio between input and output dimensions is calculated
S_R = in_rows / out_rows;
S_C = in_cols / out_cols;
% Coordinates in the image are calculated. (x,y) points are created for each point.
[cf, rf] = meshgrid(1 : out_cols, 1 : out_rows);
% x and y coordinates are multiplied by the ratio of the dimensions and then rounded to integers.
rf = rf * S_R;
cf = cf * S_C;
r = floor(rf);
c = floor(cf);
% Values outside the range in row and column coordinates are corrected.
r(r < 1) = 1;
c(c < 1) = 1;
r(r > in_rows - 1) = in_rows  - 1;
c(c > in_cols - 1) = in_cols - 1;
% The delta value in the formula is calculated
delta_R = rf - r;
delta_C = cf - c;
% For each point we want to access, the appropriate indices according to the formula are obtained
in1_ind = sub2ind([in_rows, in_cols], r, c);
in2_ind = sub2ind([in_rows, in_cols], r+1, c);
in3_ind = sub2ind([in_rows, in_cols], r, c+1);
in4_ind = sub2ind([in_rows, in_cols], r+1, c+1);
% A zero-filled image suitable for the output dimensions is created. The number of channels is set to 3 for color.
out = zeros(out_rows, out_cols, size(im, 3));
% The cast function ensures they are of the same data type.
out = cast(out, class(im));
% The interpolation formula is applied.
for idx = 1 : size(im, 3)
    % Color channels are read in sequence
    chan = double(im(:, :, idx));
    % The interpolation formula is applied
    tmp = chan(in1_ind).*(1-delta_R).*(1-delta_C) + chan(in2_ind).*(delta_R).*(1-delta_C) + chan(in3_ind).*(1-delta_R).*(delta_C) + chan(in4_ind).*(delta_R).*(delta_C);
    % It is copied to the output image.
    out(:,:,idx) = cast(tmp, class(im));
end

% Results are displayed.
figure; imshow(im); title('Original Image');
figure; imshow(out); title('Scaled Image with Bilinear Interpolation');
