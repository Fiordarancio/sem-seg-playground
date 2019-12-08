% Prepare examples for Unet 
% -------------------------------------------------------------------------
clear; close all; clc;
%--------------------------------------------------------- BEGIN USER CODE
% adjust your paths
imgInDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\originalImagesEnhanced';
imgOutDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\imagesUnet';
labInDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\originalLabelsEnhanced';
labOutDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\labelsUnet';

templatesDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\templatesColStd';
 
% flags: do you want to apply...
% ... padding and cropping?
applyPadNCrop = true;
% ... resizing?
applyResizing = false;
% ... color standardization?
applyColorStd = false;
% ... color indexing?
applyColorIdx = true;

% desired size for Unet
unetWidth = 512; 
unetHeight = 512;
% tolerances for padding & cropping
minWidth = 256;
minHeigth = 256;
arError = 1e-2; % deviation from desired aspect ratio
stride = [unetWidth, unetHeight] ./ 2; % how much pixels the sliding windows advance about
%----------------------------------------------------------- END USER CODE

prepareExamples({imgInDir, imgOutDir}, {labInDir, labOutDir}, templatesDir, ...
    [applyPadNCrop, applyResizing, applyColorStd, applyColorIdx], ...
    {segnetWidth, segnetHeight, minWidth, minHeigth, arError, percError, stride});