% Prepare images for Deeplab, that is, extracting guitars without instances
%--------------------------------------------------------------------------
clear; close all; clc;

dataPath = 'C:\Users\Ilaria\Guitar_Ilaria\dataset\EventDataset';
labInDir = fullfile(dataPath, 'rgbLabels');
labOutDir = fullfile(dataPath, 'labels');
clearDir(labOutDir);

classes = eventClasses;
labids = eventPixelLabelIDs(false);
cmap = eventColorMap;

lbds = imageDatastore(labInDir);
pxds = pixelLabelDatastore(labInDir, classes, labids);

if numel(lbds.Files) ~= numel(pxds.Files) 
    error([mfilename ': Element number mismatch'], ...
        'Number of elements in folder mus match');
end

for idx=1:numel(lbds.Files)
    [~, info] = readimage(lbds, idx);
    P = readimage(pxds, idx);
    eguitarPixels = find(P=="ElectricGuitar_Bass");
    cguitarPixels = find(P=="Classic_AcousticGuitar");
    newP = zeros(size(P));
    parfor j = 1 : size(P,1)*size(P,2)
        if ~isempty(find(eguitarPixels == j)) || ...
                ~isempty(find(cguitarPixels == j))
            newP(j) = 1;
        end
    end
    % save this one into labels
    [~, lname, lext] = fileparts(info.Filename);
    imwrite(newP, fullfile(labOutDir, [lname lext]));
end