% Preparing splits for k-fold cross-validation
%--------------------------------------------------------------------------
% K-fold cross-validation consists in splitting the whole set into k folds
% and, for each iteration, performing the evauation over the k-th set and 
% the training on the union of the k-1 other subsets. 
% In this script we just create the folds. The iteration over them will be
% done in tensorflow while training Deeplab.

clear; close all; clc;

k = input('Select number of folds: ');

% set paths and folders. We create folds as:
% -- Kfold
%   |-- fold_1
%       |-- images
%       |-- labels
%   | ...
%   |-- fold_K
%       |-- images
%       |-- labels
%--------------------------------------------------------- BEGIN USER CODE
% You can choose folds also depending on the dataset that you need
datasetPath = 'C:\Users\Ilaria\Guitar_Ilaria\dataset';
imgInDir = fullfile(datasetPath, 'imagesDeeplabPc');
labInDir = fullfile(datasetPath, 'labelsDeeplabPc');
imgOutDir= fullfile(datasetPath, 'KfoldPc');
labOutDir= fullfile(datasetPath, 'KfoldPc');
%--------------------------------------------------------- BEGIN USER CODE

if exist(imgOutDir, 'dir')
    rmpath(imgOutDir); rmdir(imgOutDir, 's');
end
if exist(labOutDir, 'dir')
    tmpath(labOutDir); rmdir(labOutDir, 's');
end
mkdir (imgOutDir); mkdir (labOutDir); 
addpath (imgOutDir); addpath (labOutDir);

% create dastastores and shuffle indexes of files
imds = imageDatastore(imgInDir);
lbds = imageDatastore(labInDir);
indx = 1 : 1 : numel(imds.Files);
indx = indx(randperm(length(indx)));

% create pixel labels just to visualize results
classes = guitarClasses()
labelIDs = guitarPixelLabelIDs(true)
cmap = guitarColorMap()
pxds = pixelLabelDatastore(labInDir, classes(), labelIDs());

% for each fold, create a folder of examples
exPerFold = floor(numel(imds.Files) / k);
residual = mod(numel(imds.Files), k);
dispPrint(['Creating ' num2str(k) ' folds with ' ...
    num2str(exPerFold) ' examples each...']);
for i = 1 : k
    images = fullfile(imgOutDir, ['fold_' num2str(i)], 'images');
    labels = fullfile(labOutDir, ['fold_' num2str(i)], 'labels');
    mkdir(images); mkdir(labels);
    
    for j = 1 : exPerFold
       [I, Iinfo] = readimage(imds, indx((i-1)*exPerFold+j));
       [L, Linfo] = readimage(lbds, indx((i-1)*exPerFold+j));
       B = labeloverlay(I, readimage(pxds, indx((i-1)*exPerFold+j)), ...
           'Colormap', cmap);
       imshow(B);
       
       [~, name, ext] = fileparts(Iinfo.Filename);
       imwrite(I, fullfile(images, [name ext]));
       [~, name, ext] = fileparts(Linfo.Filename);
       imwrite(L, fullfile(labels, [name ext]));
    end
    if i == k && residual ~= 0 % last batch, put here residuals
        disp(['Adding ' num2str(residual) ' examples in last fold...']);
        for j = 1 : residual
           [I, Iinfo] = readimage(imds, indx((i-1)*exPerFold+j));
           [L, Linfo] = readimage(lbds, indx((i-1)*exPerFold+j));
           B = labeloverlay(I, readimage(pxds, indx((i-1)*exPerFold+j)), ...
               'Colormap', cmap);
           imshow(B);

           [~, name, ext] = fileparts(Iinfo.Filename);
           imwrite(I, fullfile(images, [name ext]));
           [~, name, ext] = fileparts(Linfo.Filename);
           imwrite(L, fullfile(labels, [name ext]));
        end
    end
    disp(['Fold ' num2str(i) ' completed.']);
end






