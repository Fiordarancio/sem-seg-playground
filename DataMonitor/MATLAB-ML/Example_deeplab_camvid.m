% Semantic segmentation using Deep Learning (MATLAB TUTORIAL)
% ---------------------------------------------------------------- RECALLS
% We implement the training of a semantic segmentation network defining the
% classes we are interested in and the set of examples for it. The output
% must be a segmented image, that is an object in which all pixels are
% assigned a label, which is related to the cluster/class. 
% -------------------------------------------------------------------- NET
% We follow the example given by Matlab, downloading a Deeplab v3+ version
% already trained and designed for this purpose. Other networks to be
% tested are SegNet and U-Net (this last should be more proper for our
% purposes). Here, weights are imported from ResNet
% ---------------------------------------------------------------- DATASET
% The example uses CamVid dataset

% please update paths on 'pathSetup.m' and launch it!

% download net
pretrainedURL = 'https://www.mathworks.com/supportfiles/vision/data/deeplabv3plusResnet18CamVid.mat';
pretrainedFolder = fullfile(camviDir, 'pretrainedNetwork');
pretrainedNetwork = fullfile(pretrainedFolder, 'deeplabv3plusResnet18CamVid.mat');
% if we never downloaded the net before, just do it
if ~exist(pretrainedNetwork, 'file')
    mkdir(pretrainedFolder);
    disp('Downloading pretrained network (58 MB)...');
    websave(pretrainedNetwork, pretrainedURL);
else
    disp('Pretrained network already downloaded');
end

% download dataset
imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';

outputFolder = fullfile(camviDir, 'CamVid');
labelsZip = fullfile(outputFolder, 'labels.zip');
imagesZip = fullfile(outputFolder, 'images.zip');

if ~exist(labelsZip, 'file') || ~exist(imagesZip, 'file')
    mkdir(outputFolder);
    
    disp('Downloading CamVid dataset labels (16 MB)...'); 
    websave(labelsZip, labelURL);
    unzip(labelsZip, fullfile(outputFolder,'labels'));
    
    disp('Downloading CamVid dataset images (557 MB)...');  
    websave(imagesZip, imageURL);       
    unzip(imagesZip, fullfile(outputFolder,'images')); 
else
    disp('CamVid dataset and labels already downloaded');
end

% load images
imgDir = fullfile(outputFolder, 'images', '701_StillsRaw_full');
imds = imageDatastore(imgDir);
% display an image
I = readimage(imds, 1);
% imshow(I); title('Original image'); figure;
I = histeq(I);
imshow(I); title('High contrast image'); 

% let's take an example with its labels. A pixelLabelDatastore in an object
% that encapsulates the pixel label data and the label ID to a class name
% mapping.
% Originally, CamVid has 32 labels: in this example we are going to use 11
% of them. This means taht some labels must be grouped together, and we
% have to tell this to the network (example: 'car' groups all automotives
% classes = [
    % "Sky"; "Building"; "Pole"; "Road"; "Pavement"; ...
    % "Tree"; "SignSymbol"; "Fence"; "Car"; "Pedestrian"; "Bicyclist"
    % ]
% use all classes to see the performance
classes = camvidClasses()
	
% labelIDs = fullCamvidPixelLabelIDs()
labelIDs = camvidPixelLabelIDs()
labelDir = fullfile(outputFolder, 'labels');
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);
% superpose the label pitcture on the original one we displayed
C = readimage(pxds, 1);
cmap = camvidColorMap() % camvid has decided color to visualize labels
B = labeloverlay(I, C, 'ColorMap', cmap);
figure; imshow(B); pixelLabelColorbar(cmap, classes); 
% title('Figure with overlapping segmented mask');
% NOTE: it is possible to see pixels with no overlay. This means that those
% pixels haven't be assigned to any label. This choice is an alternative to
% the creation of a class dedicated to this kind of 'error'

% ---------------------------------------------------- COMPUTING OCURENCES
% let's count the number of pixels assigned to the class labels in order to
% do some statistics
disp('Counting occurrences...');
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

% ------------------------------------------ PREPARE DATASET AND PARAMETERS
% Prepare training, validation and test set (this tutorial uses 60% as 
% training, then 20% and 20% for validation and test
trainperc = 0.6;
valperc = 0.2;
[imdsTrain, imdsVal, imdsTest, ... 
	pxdsTrain, pxdsVal, pxdsTest] = camvidPartitionData(imds, pxds, ...
		trainperc, valperc);
numTrainImages = numel(imdsTrain.Files)
numValImages = numel(imdsVal.Files)
numTestImages = numel(imdsTest.Files)

% create the net: specify image size (should match the one on which the net
% was firstly created; specify classes
imageSize = [720, 960, 3];
numClasses = numel(classes);
lgraph = deeplabv3plusLayers(imageSize, numClasses, 'resnet18');

% check there are no 0 elements to avoid NAN results!
tbl.PixelCount(find(tbl.PixelCount == 0)) = 1;
tbl.ImagePixelCount(find(tbl.ImagePixelCount == 0)) = 1;
% balance classes with weights as we discussed before
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq % median
% assign to the pixel classifier layer: we replace the last layer,
% dedicated to classification, with the one we created for our purposes
pxLayer = pixelClassificationLayer('Name', 'labels', 'Classes', tbl.Name, 'ClassWeights', classWeights);
lgraph = replaceLayer(lgraph, 'classification', pxLayer);

% the optimization algorithm used is the Stochastic Gradient Descent with
% Momentum (SGDM) - the same I implemented for my Pitcher - and we assign
% it using trainingOptions, specifying the hyperparameters it needs
pximdsVal = pixelLabelImageDatastore(imdsVal, pxdsVal); % validation data

% DEBUG NOTE: converting RGB labels to categorical ones: in fact, the field
% 'ValidationData' of trainingOptions complains that it needs scalar labels
% instead of RGB-triplets. We are using "labels" that are images in which
% every pixel has been assigned a color (that is its label, of course) 
% pximdsVal = camvidCategoricalLabels(pximdsVal, cmap, labelDir);

miniBatchSize = 8;
maxEpochs = 30;
camvidCpDir = fullfile(camviDir, 'checkpoint');
if ~exist(camvidCpDir, 'dir')
	mkdir(camvidCpDir);
end

options = trainingOptions('sgdm', ... % next line (needed because else a neal new row is initiated)
    'LearnRateSchedule', 'piecewise', ...	%
    'LearnRateDropPeriod', 10, ...          % evrey 10 epochs...   
    'LearnRateDropFactor', 0.3, ...         % ...drop 30%
    'Momentum', 0.9, ...
    'InitialLearnRate', 1e-3, ...
    'L2Regularization', 0.005, ...
	'ValidationData', pximdsVal, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'CheckpointPath', camvidCpDir, ...	% resume from log data is training somwhy interrupts (make sure there is enough space to do that!)            
    'VerboseFrequency', 2, ...              
    'Plots', 'training-progress', ...       %
    'ValidationPatience', 4)               	%  stop if validation accuracy converges (early stopping)

% DEBUG NOTE: erased because it still complains...
%'ValidationData', pximdsVal, ...        % checking is every epoch

% data augmentation: check docs for further options
augmenter = imageDataAugmenter('RandXReflection', true, ...
    'RandXTranslation', [-10, 10], 'RandYTranslation', [-10, 10]); % pixels

% -------------------------------------------- START TRAINING 
% a final call to pixelImageDataStore will mix all together what we created
% with original examples and data augmentation
pximds = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'DataAugmentation', augmenter);
doTraining = false; % we are using pretrained values
if doTraining
    [net, info] = trainNetwork(pximds, lgraph, options);
else
%     data = load(pretrainedNetwork, 'deeplabv3plusResnet18CamVid');
%     net = data.net;
    net = load(pretrainedNetwork, 'net').net % does not complain if I load single variables
end

% --------------------------------- VISUALIZE TEST OVER ONE OR MORE IMAGES
testIdx = 35;
I = readimage(imdsTest, testIdx);
I = histeq(I);
C = semanticseg(I, net); % call for an outoput from our network
B = labeloverlay(I, C, 'Colormap', cmap, 'Transparency', 0.4);
figure; imshow(B)
pixelLabelColorbar(cmap, classes);

% compare with the ground truth. In this example, if we see a color and
% not a gray, it means that those areas are different
% DEBUG NOTE: remember that the cmap should be converted into [0,255]
% values!!!
expectedResult = readimage(pxdsTest, testIdx);
actual = uint8(C); % conversion
expected = uint8(expectedResult);
figure; imshowpair (actual, expected);

% we should notice that little object tend to be more misunderstood that
% greater ones (and this does make sense). Let's check statistically using
% the Jaccard coefficient, which is a similarity coefficient to measure the
% amount of overlap basing on the intersection-over-union (IoU) metric
iou = jaccard(C, expectedResult)*100; % disp as percentages
dic = dice(C, expectedResult)*100;
bfs = bfscore(C, expectedResult)*100;
disp(table(classes, iou, dic, bfs));

% --------------------------------------------- TEST ON THE WHOLE TEST SET
camvidTestlabels = fullfile(camviDir, 'testlabels');
if ~exist(camvidTestlabels, 'dir')
	mkdir(camvidTestlabels);
end
% launch testing on the whole testset using certain options
pxdsResults = semanticseg(imdsTest, net, ...
    'MiniBatchSize', 4, ...
    'WriteLocation', camvidTestlabels, ...
    'Verbose', true);
% evaluation
metrics = evaluateSemanticSegmentation(pxdsResults, pxdsTest, 'Verbose', true);
% see the metrics found
disp(metrics.DataSetMetrics);
% see the impact on classes in terms of accuracy coefficients
disp(metrics.ClassMetrics);

% --------------------------------------------------- SAVING THE WORKSPACE
camvidResDir = fullfile(camviDir, 'results');
if ~exist(camvidResDir, 'dir')
	mkdir(camvidResDir);
end

t = datetime();
t.Format = 'yyyy_MM_dd__HH_mm_ss';
file = strcat('net_', string(t), '_camvid.mat');
save(fullfile(camvidResDir, file));
