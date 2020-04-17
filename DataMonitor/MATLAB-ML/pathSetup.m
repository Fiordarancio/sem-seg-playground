% Setup paths that are going to be used for segmentation and/or training
%--------------------------------------------------------------------------
clear; close all; clc;

%--------------------------------------------------------- BEGIN USER CODE
% home folder
guitarDir = '/home/ilaria/Guitar_Ilaria';
% dataset folder
datasetDir = fullfile(guitarDir, 'dataset');

% path for IMAGES
% imgDir = fullfile(datasetDir, 'imagesDeeplab');
% imgDir = fullfile(datasetDir, 'imagesUnet');
imgDir = fullfile(datasetDir, 'imagesSegnet');
% path for LABELS
% labDir = fullfile(datasetDir, 'labelsDeeplab');
% labDir = fullfile(datasetDir, 'labelsUnet');
labDir = fullfile(datasetDir, 'labelsSegnet');
    
% path for training CHECKPOINTS (automatic)
guitarCpDir = fullfile(guitarDir, 'guitarNetCheckpoint');  
% path for training RESULTS (workspaces and logs)
guitarResDir= fullfile(guitarDir, 'guitarResults');
% path for produced labels 
% testlabelDir = fullfile(guitarDir, 'guitarDeeplabOutputTestLabels');   
testlabelDir = fullfile(guitarDir, 'guitarSegnetOutputTestLabels');   

% camvid folder for downloading net: since it seems I haven't found other 
% already trained networks for MATLAB, based on these architectures, we 
% continue exploiting the camvid one
camviDir = fullfile(guitarDir, 'temp_camvid');
tempUnet = fullfile(guitarDir, 'temp_unet');
%----------------------------------------------------------- END USER CODE

if ~exist(guitarResDir, 'dir')
    mkdir(guitarResDir); 
end
if ~exist(testlabelDir, 'dir')
    mkdir(testlabelDir);
end
if ~exist(guitarCpDir, 'dir')
    mkdir(guitarCpDir);
end
if ~(exist(imgDir, 'dir') || exist(labDir, 'dir'))
    disp('Warning: images/label directories do not exist!');
end

