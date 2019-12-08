% Prepare examples for Segnet 
% -------------------------------------------------------------------------
clear; close all; clc;
%--------------------------------------------------------- BEGIN USER CODE
% adjust your paths
% imgInDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\originalImagesEnhanced';
imgInDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\tmpPadImgOutDir';
imgOutDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\imagesSegnet';
% labInDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\originalLabelsEnhanced';
labInDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\tmpPadLabOutDir';
labOutDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\labelsSegnet';

templatesDir = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\templatesColStd';

% flags: do you want to apply...
% ... padding and cropping?
applyPadNCrop = false;
% ... resizing?
applyResizing = false;
% ... color standardization?
applyColorStd = true;
% ... color indexing?
applyColorIdx = true;

% desired size for SegNet
segnetWidth = 480; 
segnetHeight = 360;
% tolerances for padding & cropping
minWidth = 120;
minHeigth = 90;
arError = 1e-2; % deviation from desired aspect ratio
percError = 0.3; % percentage of minimum non-zero pixel labels
stride = [segnetWidth, segnetHeight] ./ 2;  % how much pixels the sliding 
                                            % windows advance about
%----------------------------------------------------------- END USER CODE

prepareExamples({imgInDir, imgOutDir}, {labInDir, labOutDir}, templatesDir, ...
    [applyPadNCrop, applyResizing, applyColorStd, applyColorIdx], ...
    {segnetWidth, segnetHeight, minWidth, minHeigth, arError, percError, stride});
