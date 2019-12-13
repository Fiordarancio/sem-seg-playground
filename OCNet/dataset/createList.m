function [] = createList(split, imgDir, varargin)
% Create list for a given dataset. Images and labels must have the same
% name, while images have .jpg extension and labels have the .png one.
% Launch the script from the current directory to get well-organized paths.
% Inputs:
%   - split:    name of the split
%   - imgDir:   path to images
%   - labDir:   path to annotation (optional)
%--------------------------------------------------------------------------
    narginchk(2,3);
    listDir = fullfile(pwd, 'list');
    if ~isempty(varargin)
        labDir = varargin{1};
    else
        labDir = varargin;
    end
    
    imds = imageDatastore(imgDir);
    num_img = numel(imds.Files);

    if ~isempty(labDir)
        lbds = imageDatastore(labDir);
        if num_img ~= numel(lbds.Files)
            error('Wrong number of images');
        end
    end

    % open new file
    splitListFile = fullfile(listDir, [split '.lst']);
    splitList = fopen(splitListFile, 'w+');
    for i=1 : num_img
        if ~isempty(labDir)
            fprintf(splitList, '%s %s\n', imds.Files{i}, lbds.Files{i});
        else    
            fprintf(splitList, '%s\n', imds.Files{i});
        end
    end
    fclose(splitList);
end