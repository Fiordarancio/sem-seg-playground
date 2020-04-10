function [imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = ...
    partitionGuitarData(imds, pxds, trainperc, valperc, labelIDs)
    % Partition CamVid data by randomly selecting trainperc (percentage) of the
    % data for training. The rest is divided inot valperc % for validation and 
    % the remaining is for test. Default values are 0.6, 0.2 (60%, 20%)

if trainperc+valperc >= 1.0
    disp('Training and validation sets must be less than the total');
    return;
end
    
% Set initial random state for example reproducibility.
rng(0); % set rand num gen with seed
numFiles = numel(imds.Files);
shuffledIndices = randperm(numFiles);

% Use a subset of the images for training.
N = round(trainperc * numFiles);
trainingIdx = shuffledIndices(1:N);

% Divide what's left equally
M = round(valperc * (numFiles-N));
validationIdx = shuffledIndices(N+1:N+M); 
testIdx = shuffledIndices(N+M+1:end);

% Create image datastores for training, validation and test.
trainingImages = imds.Files(trainingIdx);
validationImages = imds.Files(validationIdx);
testImages = imds.Files(testIdx);
imdsTrain = imageDatastore(trainingImages);
imdsVal = imageDatastore(validationImages);
imdsTest = imageDatastore(testImages);

% Extract class and label IDs info.
classes = pxds.ClassNames;

% Create pixel label datastores for training, validation and test and test.
trainingLabels = pxds.Files(trainingIdx);
validationLabels = pxds.Files(validationIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels, classes, labelIDs);
pxdsVal = pixelLabelDatastore(validationLabels, classes, labelIDs);
pxdsTest = pixelLabelDatastore(testLabels, classes, labelIDs);
end