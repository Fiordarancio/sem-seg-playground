% Very pretty the same of Guitar_deeplab, but we are using SegNet
% ---------------------------------------------------------------- NETWORK
% This sheet is developed modifying the CamVid example available into the
% MATLAB dicumentation. The architecture is SegNet while imported
% weights are given by VGG16 backbone.
% As a first test, we will import the network used for CamVid and we change
% classes. The "Guitar" is not present among the labels of CamVid, but this
% is a first try. 
% ---------------------------------------------------------------- DATASET
% We use a dataset which contains:
% - Zhinhan and Concert datasets, whose labels were created manually
% - a subset of Imagenet dataset, whose labels were created manually
% - a subset of Google's OpenImageV5 dataset, whose labels were selected
%	among the ones with guitars and then corrected manually
% Images are 480x360 RGB and by now categories are: 
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

clear; clc;
% please update pathSetup.m with the correct folder names!
pathSetup;

% get the VGG16 pretrained SegNet
pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/segnetVGG16CamVid.mat';
pretrainedFolder = fullfile(camviDir, 'pretrainedSegNetwork');
pretrainedSegNet = fullfile(pretrainedFolder,'segnetVGG16CamVid.mat'); 
if ~exist(pretrainedFolder,'dir')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained SegNet (107 MB)...');
    websave(pretrainedSegNet,pretrainedURL);
else
    disp('Pretrained network already downloaded');
end

% img info: segnet is trained over 480x360 (4:3) image
width = 480;
height = 360;
channels = 3;

% load images
imds = imageDatastore(imgDir);
% check image correctness
disp('Checking dataset...');
if ~checkImageCorrectness(imgDir, 'jpg', width, height, channels)
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
labelIDs = guitarPixelLabelIDs(false);

% check label correctness
if ~checkImageCorrectness(labDir, 'png', width, height, channels)
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
% NOTE: it is possible to see pixels with no overlay only if you convert
% them from categorical to colormap

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
	pxdsTrain, pxdsVal, pxdsTest] = partitionGuitarData(imds, pxds, ...
        trainperc, valperc, labelIDs);
numTrainImages = numel(imdsTrain.Files)
numValImages = numel(imdsVal.Files)
numTestImages = numel(imdsTest.Files)

% specify image size (should match the one on which the net was firstly 
% created)
imageSize = [height, width, channels];
numClasses = numel(classes);
% create SegNet layers initialized with weigths og vgg16 model
lgraph = segnetLayers(imageSize, numClasses, 'vgg16');

% balance classes with weights as we discussed before
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
for i = 1 : length(imageFreq)
    if isnan(imageFreq(i))
        imageFreq(i) = 1e-4;
    end
end
classWeights = median(imageFreq) ./ imageFreq % median
% assign to the pixel classifier layer: we replace the last layer,
% dedicated to classification, with the one we created for our purposes
% (VGG16 example needs the use of different functions than Deeplab)
pxLayer = pixelClassificationLayer('Name', 'outputLabels', 'Classes', tbl.Name, 'ClassWeights', classWeights)
lgraph = removeLayers(lgraph, 'pixelLabels'); % remove last
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph, 'softmax', 'outputLabels'); % connect(network, source_layer, dest_layer)

% by analysis of the network, we see that SegNet is FCN-like and the last
% trainable layers are a couple of decoders. We fine-tune the last
% convolutional one, keeping the same structure but setting higher
% sensibility to the learning rate variation
bkp_lgraph = lgraph; % backup if we don't want to apply training (see later)
layerTunable = lgraph.Layers(end-4:end);
layerTunable(1) = convolution2dLayer([3,3], 2, ...
    'NumChannels', 64, ...
    'Stride', [1,1], ...
    'DilationFactor', [1,1], ...
    'Padding', [1 1 1 1], ...
    'Weights', layerTunable(1).Weights, ...
    'Bias', layerTunable(1).Bias, ...
    'Name', 'decoder1_conv1_tunable', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor',10 ...
    )

% remove
lgraph = removeLayers(lgraph, ["outputLabels", "softmax", ... 
    "decoder1_relu_1", "decoder1_bn_1",  "decoder1_conv1"]);
lgraph = addLayers(lgraph, layerTunable);
lgraph = connectLayers(lgraph, 'decoder1_relu_2', 'decoder1_conv1_tunable');
% analyzeNetwork(lgraph);

% prepare validation sets and parameters for training options
pximdsVal = pixelLabelImageDatastore(imdsVal, pxdsVal); % validation data (not augmented)
miniBatchSize = 16; % outOfMemory error if 32
maxEpochs = 100;
learnRate = 5e-3;
learnRateDropPeriod = 5; 
valPatience = 10;
valFrequency = floor(numel(imdsVal.Files)/miniBatchSize);

% the optimization algorithm used is the Stochastic Gradient Descent with
% Momentum (SGDM) - the same I implemented for my Pitcher - and we assign
% it using trainingOptions, specifying the hyperparameters it needs.
% NOTE: as long as we don't train, this is optional
options = trainingOptions('sgdm', ... % next line (needed because else a neal new row is initiated)
    'LearnRateSchedule', 'piecewise', ...   % modify learning rate:
    'LearnRateDropPeriod', learnRateDropPeriod, ... % every 10 epochs...   
    'LearnRateDropFactor', 0.3, ...                 % ...drop 30% of learnRate
    'Momentum', 0.9, ...
    'InitialLearnRate', learnRate, ...      % start near to zero: only tunable layers will get an higher value
    'L2Regularization', 0.005, ...
    'ValidationData', pximdsVal, ...        % checking is every epoch
    'MaxEpochs', maxEpochs, ...             % very few epochs needed: it's just a fine tuning
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment', 'gpu', ...      % our GPU is only one 
    'CheckpointPath', guitarCpDir, ...     % resume from log data is training somwhy interrupts (make sure there is enough space to do that!)            
    'Verbose', true, ...
    'VerboseFrequency', 10, ...              
    'Plots', 'training-progress', ...       % after the training, a GUI of the plots of accuracy and loss will be displayed
    'ValidationPatience', valPatience)      % stop if validation accuracy converges (early stopping)

% data augmentation on training images
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, 'RandYReflection', true, ...
    'RandXTranslation', [-10, 10], 'RandYTranslation', [-10, 10], ...
    'RandRotation', [-45, 45], 'RandScale', [0.5, 2], ...
    'RandXShear', [-30, 30]); 

% ---------------------------------------- START TRAINING (OR FINE-TUNING)
% a final call to pixelImageDataStore will mix all together what we created
% with original examples and data augmentation
pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'DataAugmentation', augmenter);
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
%     data = load(pretrainedSegNet, 'deeplabv3plusResnet18CamVid');
%     net = data.net;
    net = load(pretrainedSegNet, 'net').net % does not complain if I load single variables
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
    'MiniBatchSize', miniBatchSize, ...
    'WriteLocation', testlabelDir, ...
    'Verbose', true);
% evaluation
metrics = evaluateSemanticSegmentation(pxdsResults,pxdsTest,'Verbose',true);
% see the metrics found
disp(metrics.DataSetMetrics);
% see the impact on classes in terms of accuracy coefficients
disp('---------------------'); disp('Class Metrics');
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

