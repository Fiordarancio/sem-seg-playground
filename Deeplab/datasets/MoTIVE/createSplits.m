% Create the splits to be used in the [train/eval]_aug test.
% Updated for motive on April 6, 2020
%--------------------------------------------------------------------------
clear; close all; clc;
dataDir = pwd;
imgDir = fullfile(dataDir, 'images_aug')
labDir = fullfile(dataDir, 'labels_aug')

trainsplitImgDir = fullfile(imgDir, 'train_images'); 
trainsplitLabDir = fullfile(labDir, 'train_labels');
evalsplitImgDir = fullfile(imgDir, 'eval_images');
evalsplitLabDir = fullfile(labDir, 'eval_labels');

if ~exist(imgDir, 'dir') || ~exist(labDir, 'dir')
    disp('No dataset from which extract anything!');
    return;
end

% reset folders
% if exist(trainsplitImgDir, 'dir')
%     rmdir(trainsplitImgDir, 's');
% end
% if exist(trainsplitLabDir, 'dir')
%     rmdir(trainsplitLabDir, 's');
% end
% if exist(evalsplitImgDir, 'dir')
%     rmdir(evalsplitImgDir, 's');
% end
% if exist(evalsplitLabDir, 'dir')
%     rmdir(evalsplitLabDir, 's');
% end
mkdir(trainsplitImgDir); mkdir(trainsplitLabDir);
mkdir(evalsplitImgDir); mkdir(evalsplitLabDir);

% shuffle and extract data
imds = imageDatastore(imgDir);
lbds = imageDatastore(labDir);

if numel(imds.Files) ~= numel(lbds.Files)
    disp('Number of images and labels mismatches!');
    return;
end
indexes = 1 : 1 : numel(imds.Files);
indexes = indexes(randperm(length(indexes)));

% read files into folder and rearrange
trainPerc = 0.8; % 80% amount of examples for training
numExs = numel(imds.Files);
disp (['Number of examples: ' num2str(numExs)]);
numTrain = round(numExs * trainPerc)
numVal = numExs - numTrain; 
for idx = 1 : numTrain
    [I, Iinfo] = readimage(imds, indexes(idx));
    [L, Linfo] = readimage(lbds, indexes(idx));
    [~, imgname, ~] = fileparts(Iinfo.Filename);
    [~, labname, ~] = fileparts(Linfo.Filename);
    if size(I,3) ~= 3
        disp(['Converting ' char(Iinfo.Filename) ' to RGB...']);
        I(:,:,2) = I(:,:,1);
        I(:,:,3) = I(:,:,1);
    end
    if size(L,3) ~= 1
        disp(['Converting label into color index (please check!)']);
        disp(L(1,1,:));
        L = L(:,:,1);
    end
    imwrite(I, fullfile(trainsplitImgDir, [imgname '.jpg']), 'jpg');
    imwrite(L, fullfile(trainsplitLabDir, [labname '.png']), 'png');
end
disp('train split aug created!');

for idx = idx : numExs
    [I, Iinfo] = readimage(imds, indexes(idx));
    [L, Linfo] = readimage(lbds, indexes(idx));
    [~, imgname, ~] = fileparts(Iinfo.Filename);
    [~, labname, ~] = fileparts(Linfo.Filename);
    if size(I,3) ~= 3
        disp(['Converting ' char(Iinfo.Filename) ' to RGB...']);
        I(:,:,2) = I(:,:,1);
        I(:,:,3) = I(:,:,1);
    end
    if size(L,3) ~= 1
        disp(['Converting label into color index (please check!)']);
        disp(L(1,1,:));
        L = L(:,:,1);
    end
    imwrite(I, fullfile(evalsplitImgDir, [imgname '.jpg']), 'jpg');
    imwrite(L, fullfile(evalsplitLabDir, [labname '.png']), 'png');
end
disp('eval split aug created!');

