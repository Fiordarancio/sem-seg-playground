% Just evaluate the maximum width and height that can be found into the
% enhanced dataset (as requested for Deeplab, see: 
%   https://github.com/tensorflow/models/issues/3939 )
%--------------------------------------------------------------------------
clear; close all; clc;
pathimg = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\originalImagesEnhanced';
imds = imageDatastore(pathimg);

max_w = 0;
max_h = 0;

while hasdata(imds)
    I = read(imds);
    if size(I,1) > max_h
        max_h = size(I, 1);
        disp(['Found new max height: ' num2str(max_h)]);
    end
    if size(I,2) > max_w
        max_w = size(I, 2);
        disp(['Found new max width: ' num2str(max_w)]);
    end
end

disp (['Greatest image dimensions are: ' num2str(max_w) ' x ' num2str(max_h)]);