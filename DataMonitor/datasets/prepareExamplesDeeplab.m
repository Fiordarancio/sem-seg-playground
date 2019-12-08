% Prepare examples for DeepLab 
% -------------------------------------------------------------------------
clear; close all; clc;
%--------------------------------------------------------- BEGIN USER CODE
% adjust your paths
dataPath = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\';
imgInDir = fullfile(dataPath, 'originalImagesEnhanced');
labInDir = fullfile(dataPath,'originalLabelsEnhanced');
imgOutDir = fullfile(dataPath, 'imagesDeeplabPc'); % only PadNCrop
labOutDir = fullfile(dataPath, 'labelsDeeplabPc');
addpath(imgInDir); addpath(labInDir); addpath(imgOutDir); addpath(labOutDir);

templatesDir = fullfile(dataPath, 'templatesColStd');

% flags: do you want to apply...
% ... padding and cropping?
applyPadNCrop = true;
% ... resizing?
applyResizing = false;
% ... color standardization?
applyColorStd = false;
% ... color indexing?
applyColorIdx = true;

% desired size for DeepLab
dplbWidth = 960; 
dplbHeight = 720;
% tolerances for padding & cropping
minWidth = 400;
minHeigth = 300;
arError = 1e-2; % deviation from desired aspect ratio
percError = 0.01; % percentage of minimum non-zero pixel labels
stride = [dplbWidth, dplbHeight] ./ 2;  % how much pixels the sliding 
                                        % windows advance about
maskCmap = [ % colors wich actually appear in the mask (range [0,1])
    zeros(1,3)
    ones(1,3)
    ];                                          
%----------------------------------------------------------- END USER CODE

prepareExamples({imgInDir, imgOutDir}, {labInDir, labOutDir}, templatesDir, ...
    [applyPadNCrop, applyResizing, applyColorStd, applyColorIdx], ...
    {dplbWidth, dplbHeight, minWidth, minHeigth, arError, percError, stride}, ...
    maskCmap);