% Arrange all available original images/labels into a proper folder
%--------------------------------------------------------------------------
clear; close all; clc;
%--------------------------------------------------------- BEGIN USER CODE
% set here the correct paths (PLEASE KEEP ")
pathDataset = "C:\Users\Ilaria\Guitar_Ilaria\dataset";
% insert here the names of the data folders you are interested in
datasets = [
    "ConcertDataset"
    "ImagenetDataset"
    "ZhinhanDataset"
    "OpenImageDatasetV5"
    "EventDataset"
    ];
%----------------------------------------------------------- END USER CODE

pathEnhancedImg = fullfile(pathDataset, "originalImagesEnhanced");
pathEnhancedLab = fullfile(pathDataset, "originalLabelsEnhanced");
if exist(pathEnhancedImg, 'dir')
    rmdir(pathEnhancedImg, 's');
end
mkdir(pathEnhancedImg); addpath(pathEnhancedImg);
if exist(pathEnhancedLab, 'dir')
    rmdir(pathEnhancedLab, 's');
end
mkdir(pathEnhancedLab); addpath(pathEnhancedLab);
    

for i = 1 : numel(datasets)
    pathsImg(i) = fullfile(pathDataset, datasets(i), "images");
    pathsLab(i) = fullfile(pathDataset, datasets(i), "labels");
end

% scan all the above folders and put all the images into another new
% folder, which also checks the correctness in terms of channels and data
% type (RGB uint8)
j = 1;
disp('Saving new enhanced dataset ...');
for i = 1 : numel(pathsImg)
    imds = imageDatastore(pathsImg(i));
    lbds = imageDatastore(pathsLab(i));
    if numel(imds.Files) ~= numel(lbds.Files)
        disp("No correspondency between elements of " );
        disp(pathsImg(i));
    else
        while hasdata(imds)
            I = uint8(read(imds));
            if size(I,3) ~= 3
%                 disp (strcat('Check image dimension: now is [', num2str(size(I)), ']'));
                I(:,:,2) = I(:,:,1);
                I(:,:,3) = I(:,:,1);
            end
            L = uint8(read(lbds));
            % labels must have 1 channel with color indexed values 
%             if size(L,3) ~= 3
% %                 disp (strcat('Check label dimension: now is [', num2str(size(L)), ']'));
%                 L(:,:,2) = L(:,:,1);
%                 L(:,:,3) = L(:,:,1);
%             end
            %------------------------- BEGIN UPDATE FOR TENSORFLOW DEEPLAB
            L = L(:,:,1); 
            if ~islogical(L)
                L = imbinarize(L);
            end
            %--------------------------- END UPDATE FOR TENSORFLOW DEEPLAB
            imwrite(I, fullfile(pathEnhancedImg, strcat(num2str(j), '.jpg')));
            imwrite(L, fullfile(pathEnhancedLab, strcat(num2str(j), '.png')));
            j = j+1;
        end
    end
end
disp(['Enhanced dataset has now ' num2str(j) ' examples.']);
