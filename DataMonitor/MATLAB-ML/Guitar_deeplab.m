% Finding Guitars - A Binary Semantic segmentation using Deep Learning
% ---------------------------------------------------------------- RECALLS
% We implement the training of a semantic segmentation network defining the
% classes we are interested in and the set of examples for it. The output
% must be a segmented image, that is an object in which all pixels are
% assigned a label, which is related to the cluster/class. 
% ---------------------------------------------------------------- NETWORK
% This sheet is developed modifying the CamVid example available into the
% MATLAB dicumentation. The architecture is Deeplab v3+ while imported
% weights are given by Resnet. Please remember that the U-NET is a
% different network architecture, so we are not going to mix that approach
% with the one used in DeepLab. Indeed, we use Segnet, which uses the
% UNet concept, to compare performances.
% As a first test, we will import the network used for CamVid and we change
% classes. The "Guitar" is not present among the labels of CamVid, but this
% is a first try. Following the state-of-the-art DeepLabV3+ implementation
% (https://github.com/tensorflow/models/tree/master/research/deeplab)
% ---------------------------------------------------------------- DATASET
% We use a dataset which contains:
% - Zhinhan and Concert datasets, whose labels were created manually
% - a subset of Imagenet dataset, whose labels were created manually
% - a subset of Google's OpenImageV5 dataset, whose labels were selected
%	among the ones with guitars and then corrected manually
% Images are 960x720 RGB and by now categories are: 
% - "Background", RGB [0,0,0], index 0
% - "Guitar", RGB [255, 255, 255], index 1 
% We will follow the same paradigm of CamVid example to write functions 
% dealing with colormaps and labels. 
% The full dataset varies about a dozen thousands of images, depending
% on the augmentation applied (padding and crop, resizing, color 
% stardardization). Other automatic image augmentations are executed
% automatically during training
% ---------------------------------------------------- CURRENT LIB VERSION
% 1.1.2

clear; close all; clc;
% please update pathSetup.m with the correct folder names!
pathSetup;

% check that we have CamVid as backbone
pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(camviDir, 'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder, 'deeplabv3plusResnet18CamVid.mat');
% if we never downloaded the net before, just do it
if ~exist(pretrainedNetwork, 'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained CamVid network (58 MB)...');
    websave(pretrainedNetwork, pretrainedURL);
else
    disp('Pretrained network already downloaded');
end

% load images
imds = imageDatastore(imgDir);
% check image correctness
disp('Checking dataset...');
if ~checkImageCorrectness(imgDir, 'jpg', 960, 720, 3)
    disp('Check your images: they have not the same properties');
    return;
else
    disp('Images have correct size and depth');
end
% display image
imIdx = 1;
I = readimage(imds, imIdx);

% let's take an example with its label mask. A pixelLabelDatastore is an 
% object that encapsulates the pixel label data and the label ID to a class
% name mapping, together with information about the files.
classes = [
    "Background"
    "Guitar"
    ];

% get IDs to be recognized on the label: since we are working with RGB data
% we expect labels to be RGB triplets (we can modify the labels with a
% proper function as we are doing here
labelIDs = guitarPixelLabelIDs();

% check label correctness
if ~checkImageCorrectness(labDir, 'png', 960, 720, 3)
    disp('Check your labels: they have not the same properties');
    return;
else
    disp('Labels have correct size and depth');
end

% load labels
pxds = pixelLabelDatastore(labDir, classes, labelIDs);
% superpose the label pitcture on the original one we displayed
C = readimage(pxds, imIdx);
cmap = guitarColorMap(); 
B = labeloverlay(I, C, 'ColorMap', cmap);
% show everything in subplot
% I = histeq(I); % high contrast
% subplot(1,2,1); imshow(I); title('Original image');
% subplot(1,2,2); 
imshow(B); pixelLabelColorbar(cmap, classes); 
title('Image with overlapping label mask');

% ---------------------------------------------------- COMPUTING OCURENCES
% let's count the number of pixels assigned to the class labels in order to
% do some statistics
disp('Counting occurrences on the whole dataset...');
tbl = countEachLabel(pxds); % display class and occurrence in pixel count
disp (tbl);

frequency = tbl.PixelCount/sum(tbl.PixelCount);
% plot a graph
figure; bar(1:numel(classes), frequency); % numel: number of array elements
xticks(1:numel(classes));
xticklabels(tbl.Name);
xtickangle(45);
ylabel('Frequency');
% we can notice that the image is highly imbalanced. That's because,
% normally, cars, road, buildings cover the greatest part of the image,
% because they are big. If we are interested in classes that are 'less
% visible' because less frequent and less big, like pedestrians and
% bicyclists, we need to leverage weighting
% NOTE: rememmber to resize the images correctly, according to the GPU
% capacity (memory)

% -------------------------------------------------------- PREPARE DATASET
% Prepare training, validation and test set (the tutorial uses 60% as 
% training, then 20% and 20% for validation and test)
trainperc = 0.6;
valperc = 0.2;
[imdsTrain, imdsVal, imdsTest, ...
	pxdsTrain, pxdsVal, pxdsTest] = partitionGuitarData(imds, pxds, trainperc, valperc);
numTrainImages = numel(imdsTrain.Files)
numValImages = numel(imdsVal.Files)
numTestImages = numel(imdsTest.Files)

% create the net: specify image size (should match the one on which the net
% was firstly created) specify classes
imageSize = [720, 960, 3];
numClasses = numel(classes);
lgraph = deeplabv3plusLayers(imageSize, numClasses, 'resnet18');

% balance classes with weights as we discussed before
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq % median
% to perform fine tuning, we have to have some trainable layer to the
% Deeplab since it has no one left of this type at its end, instead of
% SegNet has. So, we try by first to add a fully connected layer before the
% softmax-out one. Moreover, we replace the final classification labels
% with our one. 
% DEBUG NOTE: Matlab does not recognize correctly the size of
% classification layer. Moreover, DeepLab shoul have a single checkpoint
% from which is good to train the model, according to the paper on Deeplab
% https://github.com/tensorflow/models/tree/master/research/deeplab
pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'ClassWeights', classWeights);
lgraph = replaceLayer(lgraph, 'classification', pxLayer);
% fcLayer = fullyConnectedLayer(numClasses, ... % outputsize
%     'Name', 'finetuning_fc', ...
%     'NumInputs', imageSize(1)*imageSize(2)*imageSize(3), ...
%     'WeightLearnRateFactor',10, ...
%     'BiasLearnRateFactor',10);
% smLayer = softmaxLayer('Name', 'softmax-out');
% lgraph = removeLayers(lgraph, 'classification');
% lgraph = removeLayers(lgraph, 'softmax-out');
% lgraph = addLayers(lgraph, fcLayer);
% lgraph = addLayers(lgraph, smLayer);
% lgraph = addLayers(lgraph, pxLayer);
% lgraph = connectLayers(lgraph, 'dec_crop2', 'finetuning_fc');
% lgraph = connectLayers(lgraph, 'finetuning_fc', 'softmax-out');
% lgraph = connectLayers(lgraph, 'softmax-out', 'labels');
analyzeNetwork(lgraph);

% prepare validation sets and parameters for training options
pximdsVal = pixelLabelImageDatastore(imdsVal, pxdsVal); % validation data (not augmented)
miniBatchSize = 8; %8
maxEpochs = 20; %2
valFrequency = floor(numel(imdsVal.Files)/miniBatchSize);

% the optimization algorithm used is the Stochastic Gradient Descent with
% Momentum (SGDM) - the same I implemented for my Pitcher - and we assign
% it using trainingOptions, specifying the hyperparameters it needs.
% NOTE: as long as we don't train, this is optional
options = trainingOptions('sgdm', ... % next line (needed because else a neal new row is initiated)
    'LearnRateSchedule', 'piecewise', ...   % modify learning rate:
    'LearnRateDropPeriod', 10 , ...          % every 10 epochs...   
    'LearnRateDropFactor', 0.3, ...         % ...drop 30%
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...           % start near to zero: only tunable layers will get an higher value
    'L2Regularization', 0.005, ...
    'ValidationData', pximdsVal, ...        % checking is every epoch
    'MaxEpochs', maxEpochs, ...             % very few epochs needed: it's just a fine tuning
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu', ...      % our GPU is only one 
    'CheckpointPath', guitarCpDir, ...      % resume from log data is training somwhy interrupts (make sure there is enough space to do that!)            
    'Verbose', true, ...
    'VerboseFrequency', 10, ...             % write the checkpoint result on console every 10 iterations    
    'Plots', 'training-progress', ...       % after the training, a GUI of the plots of accuracy and loss will be displayed
    'ValidationPatience', 10) %4            % stop if validation accuracy converges (early stopping)

% data augmentation on training images
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, 'RandYReflection', true, ...
    'RandXTranslation', [-10, 10], 'RandYTranslation', [-10, 10], ...
    'RandRotation', [-90, 90], 'RandScale', [0.5, 2], ...
    'RandXShear', [-30, 30]); 

% ---------------------------------------- START TRAINING (OR FINE-TUNING)
% a final call to pixelImageDataStore will mix all together what we created
% with original examples and data augmentation
pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);
if input('Start fine-tuning? [y/n] ', 's') == 'y' % we are using pretrained values
    doTraining = true;
else
    doTraining = false;
end

if doTraining
    disp('Fine tuning started...');
    [net, info] = trainNetwork(pximds, lgraph, options);
%     [net, info] = trainNetwork(pximds, bkp_lgraph, options);
else
    disp('Loading pretrained network...');
    disp('Warning: all layers here are built according to Camvid needs!');
%     data = load(pretrainedNetwork, 'deeplabv3plusResnet18CamVid');
%     net = data.net;
    net = load(pretrainedNetwork, 'net').net % does not complain if I load single variables
end

% --------------------------------- VISUALIZE TEST OVER ONE OR MORE IMAGES
for imIdx = 1 : 5
    testImage = readimage(imdsTest, imIdx);
    expectedResult = readimage(pxdsTest, imIdx);
    actualResult = guitarEvaluateResult(net, testImage, expectedResult, classes, cmap);
end

% --------------------------------------------- TEST ON THE WHOLE TEST SET
% launch testing on the whole testset using certain options
pxdsResults = semanticseg(imdsTest, net, ...
    'MiniBatchSize', 4, ...
    'WriteLocation', testlabelDir, ...
    'Verbose', true);
% evaluation
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest, 'Verbose', true);
% see the metrics found
disp(metrics.DataSetMetrics);
% see the impact on classes in terms of accuracy coefficients
disp(metrics.ClassMetrics);


% --------------------------------------------------- SAVING THE WORKSPACE
t = datetime();
t.Format = 'yyyy-MM-dd_HH-mm-ss';
netName = 'segnet_finetuned';
% save datastores before (sometimes, load crashes on loading them)
filename = [netName '-' char(t) '-datastores.mat'];
filename = fullfile(guitarResDir, filename);
dispPrint('Saving datastores...');
save(filename, ...
    'imds', ...     % main imageDatastore
    'imdsTest', ... % test split
    'imdsTrain', ...% train split
    'imdsVal', ...  % val split
    'pxds', ...     % main pixelLabelDatastore
    'pxdsTest', ... % test split  
    'pxdsTrain', ...% train split
    'pxdsVal', ...  % val split
    'pximds', ...   % examples imdsTrain + pxdsTrain using augmenter
    'pximdsVal' ... % examples imdsVal + pxdsVal
    );
% remove these variables and save the rest
clear imds imdsTest imdsTrain imdsVal 
clear pxds pxdsTest pxdsTrain pxdsVal pximds pximdsVal
filename = [netName '-' char(t) '-workspace.mat'];
filename = fullfile(guitarResDir, filename);
disp('Saving workspace...');
save(filename);
