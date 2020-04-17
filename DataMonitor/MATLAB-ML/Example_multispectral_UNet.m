% Semantic Segmentation of Multispectral Images using UNet
%--------------------------------------------------------------------------
% Find out the MATLAB tutorial
% https://es.mathworks.com/help/deeplearning/examples/ ...
% multispectral-semantic-segmentation-using-deep-learning.html

clear; close all; clc;

% download dataset
tempDir = '/home/ilaria/Guitar_Ilaria/temp_unet';
datasetDir = fullfile(tempDir, 'HamlinbeachMSIData');
resultDir = fullfile(tempDir, 'trainResultDir');
if ~exist(tempDir, 'dir')
    mkdir(tempDir); addpath(tempDir);
end
if ~exist(resultDir, 'dir')
    mkdir(resultDir); addpath(resultDir);
end
data = fullfile(datasetDir, 'rit18_data.mat')
if ~exist(datasetDir, 'dir')
    disp('Downloading HamlinBeachMSIData dataset (~3.0 GB)...');
    mkdir(datasetDir); addpath(datasetDir);
    url = 'http://www.cis.rit.edu/~rmk6217/rit18_data.mat';
    websave(data, url);
else
    disp('Dataset already downloaded');
end

% download pretrained network
pretrainedNetDir = fullfile(tempDir, 'pretrainedUNet');
if ~exist(pretrainedNetDir, 'dir')
    disp('Downloading pretrained UNet...');
    mkdir(pretrainedNetDir); addpath(pretrainedNetDir);
    url = 'https://www.mathworks.com/supportfiles/vision/data/multispectralUnet.mat';
    obj = fullfile(pretrainedNetDir, 'multispectralUnet.mat');
    websave(obj, url);
else
    disp('Pretrained network already downloaded');
end

% load and inspect
load(data);
whos train_data val_data test_data
% data have channel at the beginning: we must switch them to visualize
train_data = switchChannelsToThirdPlane(train_data);
val_data   = switchChannelsToThirdPlane(val_data);
test_data  = switchChannelsToThirdPlane(test_data);
figure
% montage displays multiple image frames as a rectangular montage.
% Parameters are size, thumbnail size, parent (axis to which any other
% image is adapted on), indices (which frames are included)
montage(... % display RGB (first 3 channels)
    {histeq(train_data(:,:,[3 2 1])), ...
    histeq(val_data(:,:,[3 2 1])), ...
    histeq(test_data(:,:,[3 2 1]))}, ...
    'BorderSize',10,'BackgroundColor','white')
title('RGB Component of Training (Left), Validation (Center) and Test Image (Right)')
figure
montage(... % display infrared channel (from 4 to 6)
    {histeq(train_data(:,:,4)), ...
    histeq(train_data(:,:,5)), ...
    histeq(train_data(:,:,6))}, ...
    'BorderSize',10,'BackgroundColor','white')
title('Infrared channels of Training Images (from 1 to 3)')
figure
montage(...
    {train_data(:,:,7), ... % display the mask: the valid segmentation part of the images
    val_data(:,:,7), ...
    test_data(:,:,7)}, ...
    'BorderSize',10,'BackgroundColor','white')
title('Mask of Training Image (Left), Validation Image (Center), and Test Image (Right)')

% just display classes
disp(classes);

% create class names
classNames = [ ...
    "RoadMarkings", ...
    "Tree", ...
    "Building", ...
    "Vehicle", ...
    "Person", ...
    "LifeguardChair", ...
    "PicnicTable", ...
    "BlackWoodPanel",...
    "WhiteWoodPanel", ...
    "OrangeLandingPad", ...
    "Buoy", ...
    "Rocks",...
    "LowLevelVegetation", ...
    "Grass_Lawn", ...
    "Sand_Beach", ...
    "Water_Lake", ...
    "Water_Pond", ...
    "Asphalt" ...
    ]; 

% overlay labels using the colorbar with jet
N = numel(classNames);
cmap = jet(N);
B = labeloverlay(histeq(train_data(:,:,4:6)), train_labels, ...
    'Transparency', 0.8, 'Colormap', cmap);
figure
imshow(B)
title('Training labels');
% using operator min : step : max, meaning that it creates a vector whose
% first element is min and subsequent elements proceed by adding step at
% each element until the next one is <= max. The number of elements in the
% vector is obtained by consequence
ticks = 1/(N*2) : 1/N : 1; 
colorbar('TickLabels', cellstr(classNames), 'Ticks', ticks, ...
    'TickLength', 0, 'TickLabelInterpreter', 'none')
colormap(cmap)

% divide images from labels in order to save them differently
trainData = fullfile(tempDir, 'train_data.mat');
trainLabs = fullfile(tempDir, 'train_labels.png');
save(trainData, 'train_data');
imwrite(train_labels, trainLabs);

% use Random Patch Extraction Datastore for training: this kind of
% datastore is very useful to avoid running out of memory during training.
% It extracts random patches of known size from an example. Here, our
% images and labels are not in standard format
imds = imageDatastore(trainData, ...
    'FileExtensions', '.mat', 'ReadFcn', @matReader);
pixelLabeIds = 1 : N; % just index labels
pxds = pixelLabelDatastore(trainLabs, classNames, pixelLabeIds);
rpeds = randomPatchExtractionDatastore(imds, pxds, [256,256], ...
    'PatchesPerImage', 16000);
% exploit this datastore to provide random patches for the training
inputBatch = preview(rpeds);
disp(inputBatch);

%------------------------------------------------------ PREPARE A NEW UNET
% This example uses a variation of the UNet: classical structure uses
% pooling layers to decrease the resolution of the image, then on the
% contrary the resolution is restored by another series of convolutional
% layers. THIS fact gives the name 'U': because it's symmetric and in the
% end we return from where we started.
% Here, the modification involves the use of zero-padding at convolutions
% in order to adappt the images to the same size. the mask helps
% recognizing the relevant sections. 
inputTileSize = [256, 256, 6]; 
lgraph = createUnet(inputTileSize);
disp(lgraph.Layers);
analyzeNetwork(lgraph);

% training options, as known in other examples
initialLearningRate = 5e-2;
maxEpochs = 1; %150;
miniBatchSize = 16;
L2reg = 1e-4; % L2 regularization (recover info)

options = trainingOptions('sgdm', ...
    'InitialLearnRate',initialLearningRate, ...
    'Momentum',0.9,...
    'L2Regularization',L2reg,...
    'MaxEpochs',maxEpochs,...
    'MiniBatchSize',miniBatchSize,...
    'LearnRateSchedule','piecewise',...    
    'Shuffle','every-epoch',...
    'GradientThresholdMethod','l2norm',...
    'GradientThreshold',0.05, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'VerboseFrequency', 20)

%----------------------------------- START TRAINING OR LOAD PRETRAINED NET
% DEBUG NOTE: cannot run on local windows due to the lack of GPU memory
% please run on the server...
answer = input('Train the new U-net from scratch? [y/n] ', 's');
if strcmp(answer, 'y')
    disp('Starting training....');
    doTraining = true;
else
    disp('Using pretrained network.');
    doTraining = false;
end

if doTraining
    t = datetime();
    t.Format = 'dd-MM-yyyy-HH-mm-SS';
    % train
    [net, info] = trainNetwork(rpeds, lgraph, options);
    % save results
    resFile = fullfile(resultDir, ...
        ['multispectralUnet-ex-' char(t) '-Epoch-' num2str(maxEpochs) '.mat']);
    save(resFile, 'net', 'options');
else
    load(fullfile(pretrainedNetDir, 'multispectralUnet.mat'));
end

%------------------------------------------------------------ TEST RESULTS
predictedPatchSize = [1024 1024]; % why this?
segmentedImage = segmentImage(val_data, net, predictedPatchSize);
% isolate the mask (take only the non-zero-values)
segmentedImage = uint8(val_data(:,:,7) ~= 0) .* segmentedImage;
figure
subplot(1,2,1)
imshow(segmentedImage, [])
title('Segmented Image')
% adjust the noise with a post-processing (median filter)
segmentedImage = medfilt2(segmentedImage, [7,7]);
subplot(1,2,2)
imshow(segmentedImage, []);
title('Segmented Image with noise removed');
% and finally, the overlay
B = labeloverlay(histeq(val_data(:,:,[3,2,1])), segmentedImage, ...
    'Transparency', 0.65, 'Colormap', cmap);
figure
imshow(B)
title('Labeled validation image')
colorbar('TickLabels', cellstr(classNames), 'Ticks', ticks, ...
    'TickLength', 0, 'TickLabelInterpreter', 'none');
colormap(cmap)

% save
imwrite(segmentedImage, fullfile(resultDir, 'results.png'));
imwrite(val_labels, fullfile(resultDir, 'gtruth.png'));

%-------------------------- EVALUATE THE QUALITY OF SEGMENTATION: ACCURACY
% let's use the common function for evaluation, as we did in other examples
pxdsResult = pixelLabelDatastore(fullfile(resultDir, 'results.png'), ...
    classNames, pixelLabeIds);
pxdsGTruth = pixelLabelDatastore(fullfile(resultDir, 'gtruth.png'), ...
    classNames, pixelLabeIds);

ssm = evaluateSemanticSegmentation(pxdsResult, pxdsGTruth, ...
    'Metrics', 'global-accuracy');

% evaluate incidence of a certain label: in this example we focus on
% vegetarion (datasets were built to detect deforestation)
vegetationClassIds = uint8([2,13,14]);
% find pixeld with the aforementioned label in the segmented mask
vegetationPixels = ismember(segmentedImage(:), vegetationClassIds);
validPixels = (segmentedImage ~= 0); 

numVegetationPixels = sum(vegetationPixels(:)); % sum because...
numValidPixels = sum(validPixels(:));

percentageVegetationCoverage = numVegetationPixels*100/numValidPixels;
fprintf('Percentage of vegetation %3.2f%%\n', percentageVegetationCoverage);


