% Finding Guitars pt.3 - A Binary Semantic segmentation using Deep Learning
%--------------------------------------------------------------------------
% In this script we follow the approach shown in multispectral semantic
% segmanetation to solve the problem already discussed with Deeplab and
% SegNet. The code is freely adapted by the homonymous MATLAB tutorial
% ---------------------------------------------------------------- DATASET
% RGB multilabeled images
% ---------------------------------------------------- CURRENT LIB VERSION
% 1.1.2
% In the very first version, anything is HARD CODED

clear; close all; clc;

% connect to dataset using pathSetup
pathSetup;
addpath('/home/ilaria/MATLAB/examples');

% download pretrained network
pretrainedNetDir = fullfile(tempUnet, 'pretrainedUNet');
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
classes = guitarClasses()
cmap = guitarColorMap();
labelIDs = guitarPixelLabelIDs();
imds = imageDatastore(imgDir);
% lbds = imageDatastore(labDir);
pxds = pixelLabelDatastore(labDir, classes, labelIDs);

imIdx = 1;
Img = readimage(imds,imIdx);
% L = readimage(lbds,imIdx);
Lab = readimage(pxds,imIdx);
B = labeloverlay(Img, Lab, 'Transparency', 0.7, 'Colormap', cmap);
figure
imshow(B); title('Example of image with overlaid mask'); 
pixelLabelColorbar(cmap, classes);

% all datas are in datastore... no need to montage
whos Img Lab

% subdivide the dataset in train/val/test
trainperc = 0.6;
valperc = 0.2;
[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = ...
    partitionGuitarData(imds, pxds, trainperc, valperc);
numTrainImg = numel(imdsTrain.Files)
numValImg = numel(imdsVal.Files)
numTestImg = numel(imdsTest.Files)

% use Random Patch Extraction Datastore for training: this kind of
% datastore is very useful to avoid running out of memory during training.
% When you use a randomPatchExtractionDatastore as a source of training 
% data, the datastore extracts multiple random patches from each image for 
% each epoch, so that each epoch uses a slightly different data set. The 
% actual number of training patches at each epoch is the number of training
% images multiplied by PatchesPerImage. The image patches are not stored in
% memory. 
patchSize = [256, 256];
rpeds = randomPatchExtractionDatastore(imdsTrain, pxdsTrain, ...
    patchSize, 'PatchesPerImage', 10);
% read and view the first
inputBatch = preview(rpeds);
disp('Random input patch is');
disp(inputBatch);
% try to use the same for evaluation
rpedsVal = randomPatchExtractionDatastore(imdsVal, pxdsVal, ...
    patchSize, 'PatchesPerImage', 1);

%------------------------------------------------------ PREPARE A NEW UNET
% This example uses a variation of the UNet: classical structure uses
% pooling layers to decrease the resolution of the image, then on the
% contrary the resolution is restored by another series of convolutional
% layers. THIS fact gives the name 'U': because it's symmetric and in the
% end we return from where we started.
% Here, the modification involves the use of zero-padding at convolutions
% in order to adappt the images to the same size. the mask helps
% recognizing the relevant sections. 
inputTileSize = [256, 256, 3]; 
lgraph = createUnet(inputTileSize);
disp(lgraph.Layers);
analyzeNetwork(lgraph);
% replace with our layers
newFinalConv = convolution2dLayer([1,1], 2, 'Name', 'NewFinalConvolutionLayer');
lgraph = replaceLayer(lgraph, 'Final-ConvolutionLayer', newFinalConv);
newFinalSoftMax = softmaxLayer('Name', 'NewFinalSoftmaxLayer');
lgraph = replaceLayer(lgraph, 'Softmax-Layer', newFinalSoftMax);
newClassifier = pixelClassificationLayer('Classes', classes, ...
    'Name', 'NewFinalClassificationLayer');
lgraph = replaceLayer(lgraph, 'Segmentation-Layer', newClassifier);
% display what happened
disp(lgraph.Layers);
analyzeNetwork(lgraph);

% training options, as known in other examples
initialLearningRate = 5e-2; %5e-2
maxEpochs = 50; %150
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
    'VerboseFrequency', 100, ...
    'ValidationData', rpedsVal, ...
    'ExecutionEnvironment', 'gpu', ...
    'ValidationPatience', 10, ...
    'LearnRateDropPeriod', 10, ...
    'LearnRateDropFactor', 0.25, ...
    'CheckpointPath', guitarCpDir)
%     

% % barely from the example
% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',initialLearningRate, ...
%     'Momentum',0.9,...
%     'L2Regularization',L2reg,...
%     'MaxEpochs',maxEpochs,...
%     'MiniBatchSize',miniBatchSize,...
%     'LearnRateSchedule','piecewise',...    
%     'Shuffle','every-epoch',...
%     'GradientThresholdMethod','l2norm',...
%     'GradientThreshold',0.05, ...
%     'Plots','training-progress', ...
%     'ExecutionEnvironment', 'gpu', ...
%     'VerboseFrequency', 20, ...
%     'CheckpointPath', guitarCpDir)

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
    t.Format = 'dd-MM-yyyy-HH-mm-ss';
    % train
    [net, info] = trainNetwork(rpeds, lgraph, options);
%     % save results
%     resFile = fullfile(guitarResDir, ...
%         ['multispectralUnet-' char(t) '-workspace.mat']);
%     save(resFile) %, 'net', 'options');
else
    load(fullfile(pretrainedNetDir, 'multispectralUnet.mat'));
end

%-------------------------------------- TEST RESULTS (ON A VALIDATION IMG)
predictedPatchSize = [512, 512]; % suppose this is the maximum size
imIdx = 1;
valImg = readimage(imdsVal, imIdx);
whos valImg
segmentedImg = segmentImage(valImg, net, predictedPatchSize);
% % isolate the mask (take only the non-zero-values)
% segmentedImage = uint8(val_data(:,:,7) ~= 0) .* segmentedImage;
figure
subplot(2,2,1)
imshow(histeq(segmentedImg))
title('Segmented Image')
% adjust the noise with a post-processing (median filter)
segmentedImg = medfilt2(segmentedImg, [7,7]);
subplot(2,2,3)
imshow(segmentedImg, []);
title('Segmented Image with noise removed');
% and finally, the overlay
B = labeloverlay(valImg, segmentedImg, 'Transparency', 0.65, 'Colormap', cmap);
subplot(2,2,[2,4]);
imshow(B)
title('Labeled validation image'); pixelLabelColorbar(cmap, classes);

% evaluate incidence of a certain label: the guitar
guitarClassIds = uint8(2);
% find pixels with the aforementioned label in the segmented mask
guitarPixels = ismember(segmentedImg(:), guitarClassIds);
validPixels = (segmentedImg ~= 0); 

numGuitarPixels = sum(guitarPixels(:)); % sum because...
numValidPixels = sum(validPixels(:));

guitarCoveragePerc = numGuitarPixels/numValidPixels;
fprintf('Percentage of guitar %3.2f%%\n', guitarCoveragePerc);

% save
imwrite(segmentedImg, fullfile(guitarResDir, ...
    ['guitar_unet_valresult_' num2str(imIdx) '.png']));
imwrite(readimage(pxdsVal, imIdx), fullfile(guitarResDir, ...
    ['guitar_unet_valgtruth_' num2str(imIdx) '.png']));

%-------------------------- EVALUATE THE QUALITY OF SEGMENTATION: ACCURACY
% prepare a whole datastore with results to be compared with the val set.
% Here we are using a different function for evaluation... check the code
% for further inspection
for imIdx = 1 : numel(imdsVal.Files)
    valImg = readimage(imdsVal, imIdx);
    segmentedImg = segmentImage(valImg, net, predictedPatchSize);
    imwrite(segmentedImg, fullfile(testlabelDir, ...
        ['pixelLabel_' num2str(imIdx) '.png']));
end
% let's use the common function for evaluation, as we did in other examples
pxdsRes = pixelLabelDatastore(testlabelDir, classes, labelIDs);
ssm = evaluateSemanticSegmentation(pxdsRes, pxdsVal, ...
    'Metrics', 'global-accuracy');

disp(ssm);

%------------------------------------------------------ SAVE THE WORKSPACE
% % t = datetime();
% % t.Format = 'yyyy-MM-dd_HH-mm-ss';
% netName = 'mustispectralUnet';
% % save datastores before (sometimes, load crashes on loading them)
% filename = [netName '-' char(t) '-datastores.mat'];
% filename = fullfile(guitarResDir, filename);
% dispPrint('Saving datastores...');
% save(filename, ...
%     'imds', ...     % main imageDatastore
%     'imdsTest', ... % test split
%     'imdsTrain', ...% train split
%     'imdsVal', ...  % val split
%     'pxds', ...     % main pixelLabelDatastore
%     'pxdsTest', ... % test split  
%     'pxdsTrain', ...% train split
%     'pxdsVal', ...  % val split
%     'pximds', ...   % examples imdsTrain + pxdsTrain using augmenter
%     'pximdsVal' ... % examples imdsVal + pxdsVal
%     );
% % remove these variables and save the rest
% clear imds imdsTest imdsTrain imdsVal 
% clear pxds pxdsTest pxdsTrain pxdsVal pximds pximdsVal
% filename = [netName '-' char(t) '-workspace.mat'];
% filename = fullfile(guitarResDir, filename);
% disp('Saving workspace...');
% save(filename);