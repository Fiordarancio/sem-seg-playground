% Prepare examples from Motive dataset, for Tensorflow DeepLab.
% Details of dataset:
%   - classes: 35
%   - ignore label: [0,0,0], index 0
%   - number of elements: 52
% The Tensorflow implementation of Deeplab is invariant w.r.t. the size of
% the images, as we can see from the experiments made with PASCAL VOC.
% Hence, the following requirements are needed only if a size-homogeneous
% dataset becomes required:
%   - desired dimension: 960 x 720
%   - desired augmentation: pad&crop, indexing
% -------------------------------------------------------------------------
clear; close all; clc;
%--------------------------------------------------------- BEGIN USER CODE
% adjust your paths
dataPath = 'C:\Users\Ilaria\Guitar_Ilaria\dataset';
% imgInDir = fullfile(dataPath, 'MoTIVEDataset', 'motiveImages');
% labInDir = fullfile(dataPath,'MoTIVEDataset', 'rgbLabels');
% imgOutDir = fullfile(dataPath, 'imagesMoTIVE'); 
% labOutDir = fullfile(dataPath, 'labelsMoTIVE');
imgInDir = fullfile(dataPath, 'MoTIVEDataset', 'test', 'original');
labInDir = imgInDir;
imgOutDir = fullfile(dataPath, 'MoTIVEDataset', 'test', '1440720'); 
labOutDir = fullfile(dataPath, 'MoTIVEDataset', 'test', 'trash');
%----------------------------------------------------------- END USER CODE
clearDir(imgOutDir); clearDir(labOutDir);

%% Optional requirements
%--------------------------------------------------------- BEGIN USER CODE
templatesDir = fullfile(dataPath, 'templatesColStd');
% flags: do you want to apply...
% ... padding and cropping?
applyPadNCrop = true;
% ... resizing?
applyResizing = false;
% ... color standardization?
applyColorStd = false;
% ... color indexing FOR 1 LABEL?
applyColorIdx = false;

% desired size for DeepLab
mtveWidth = 1440; % * 1.5; 
mtveHeight = 1080; % * 1.5;
% tolerances for padding & cropping
minWidth = 400;
minHeigth = 300;
arError = 0.0; % deviation from desired aspect ratio
percError = 0.5; % percentage of minimum non-zero pixel labels
stride = [mtveWidth, mtveHeight] ./ 2;  % how much pixels the sliding 
                                        % windows advance about   
maskCmap = motiveColorMap;                                        
%----------------------------------------------------------- END USER CODE
%% custom resizing
tmpImgOutDir = fullfile(dataPath, 'tmpImgOutDir'); clearDir(tmpImgOutDir);
tmpLabOutDir = fullfile(dataPath, 'tmpLabOutDir'); clearDir(tmpLabOutDir);
prepareResizedDataset_v2(imgInDir, tmpImgOutDir, labInDir, tmpLabOutDir, ...
    mtveWidth, mtveHeight, false);
imgInDir = tmpImgOutDir; labInDir = tmpLabOutDir; 
%%
prepareExamples({imgInDir, imgOutDir}, {labInDir, labOutDir}, templatesDir, ...
    [applyPadNCrop, applyResizing, applyColorStd, applyColorIdx], ...
    {mtveWidth, mtveHeight, minWidth, minHeigth, arError, percError, stride}, ...
    maskCmap);
%%
labInDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\labelsMoTIVE_960720\train';
labOutDir = labInDir;
prepareIndexedLabels(labInDir, labOutDir, motivePixelLabelIDs(false));