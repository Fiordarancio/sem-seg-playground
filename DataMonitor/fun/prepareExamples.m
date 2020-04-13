function [] = prepareExamples(imgDirs, labDirs, flags, varargin) % padOptions, templatesDir, cmap)
% FUTURE VERSION: add varargin{:}
% imgDirs:      input and output directories for images
% labDirs:      input and output directories for labels
% flags:        required flags to apply augmentations are 
%               1) PADDING & CROPPING
%               2) RESIZING
%               3) COLOR STANDARDIZATION
%               4) CONVERSION OF LABELS TO COLOR INDEXES
% VARARGIN
% padOptions:   cell array for the Padding & Crop function (see
%               preparePaddingDataset function)
% templateDir:  path to templates for color standardization
% cmap          RGB cmap to transform each color into an index
    narginchk(2,6);
    % check folder validity
    if numel(imgDirs) ~= 2 || numel(labDirs) ~= 2
        disp('Invalid directories: must have input and output directories for images and labels.');
        disp('Type help for more information');
        return;
    end
    imgInDir = imgDirs{1};
    imgOutDir = imgDirs{2};
    labInDir = labDirs{1};
    labOutDir = labDirs{2};
    % check flag validity
    if nargin < 4 || numel(flags) ~= 4
        disp ('Invalid number of flags. Must be 4.');
        disp ('Type help for more information');
        return;
    end
    applyPadNCrop = flags(1);
    applyResizing = flags(2);
    applyColorStd = flags(3);
    applyColorIdx = flags(4);
    % check options validity
    padOptions = varargin{1};
    if applyPadNCrop
        if numel(padOptions) ~= 7
            disp('Invalid options for Pad&Crop method. Must be:');
            disp('- desired width and height');
            disp('- min acceptable width and height');
            disp('- min deviation from the desired aspect ratio');
            disp('- min amount of non-zero pixel labels (percentage)');
            disp('- stride for sliding windows');
            return;
        end
            desWidth =  padOptions{1};
            desHeight = padOptions{2};
            minWidth =  padOptions{3};
            minHeigth = padOptions{4};
            arError =   padOptions{5};
            percError = padOptions{6};
            stride =    padOptions{7};
    end
    % check templates dir validity
    templatesDir = varargin{2};
    if applyColorStd 
        if ~exist(templatesDir, 'dir')
            disp('Invalid directory for templates.');
            return;
        end
    end
    % check cmap validity
    cmap = varargin{3};
    if applyColorIdx
%         if nargin < 6 || size(cmap, 2) ~= 3
        if size(cmap, 2) ~= 3
            disp('Invalid cmap for color index. Must be a m-by-3 matrix');
            return;
        end
    end
    
    %---------------------------------------------------- BEGIN PROCESSING
    if applyPadNCrop
        dispPrint('Applying padding & crop to dataset...');
        % create temporary output, if there are other steps
        if applyResizing || applyColorStd || applyColorIdx
            tmpPadImgOutDir = fullfile(fileparts(imgOutDir), 'tmpPadImgOutDir');
            tmpPadLabOutDir = fullfile(fileparts(labOutDir), 'tmpPadLabOutDir');
            if exist(tmpPadImgOutDir, 'dir') || exist(tmpPadLabOutDir, 'dir')
                rmdir(tmpPadImgOutDir, 's'); rmdir(tmpPadLabOutDir, 's');
            end
            mkdir(tmpPadImgOutDir); mkdir(tmpPadLabOutDir);
        else
            tmpPadImgOutDir = imgOutDir;
            tmpPadLabOutDir = labOutDir;
        end
        % apply method
        preparePaddingDataset(imgInDir, tmpPadImgOutDir, ...
                labInDir, tmpPadLabOutDir, ...
                desWidth, desHeight, minWidth, minHeigth, ...
                arError, percError, stride);
        % prepare in path for the following
        imgInDir = tmpPadImgOutDir;
        labInDir = tmpPadLabOutDir;
        % inform about where the last step successfully completed
        dispPrint('PadNCrop completed');
        disp(['Images saved at: ' imgInDir]);
        disp(['Labels saved at: ' labInDir]);
    end
    % resizing
    if applyResizing
        dispPrint('Applying resizing to dataset...');
        if applyColorStd || applyColorIdx
            tmpResImgOutDir = fullfile(fileparts(imgOutDir), 'tmpResImgOutDir');
            tmpResLabOutDir = fullfile(fileparts(labOutDir), 'tmpResLabOutDir');
            if exist(tmpResImgOutDir, 'dir') || exist(tmpResLabOutDir, 'dir')
                rmdir(tmpResImgOutDir, 's'); rmdir(tmpResLabOutDir, 's');
            end
            mkdir(tmpPadImgOutDir); mkdir(tmpPadLabOutDir);
        else
            tmpResImgOutDir = imgOutDir;
            tmpResLabOutDir = labOutDir;
        end

        prepareResizedDataset(imgInDir, tmpResImgOutDir, labInDir, tmpResLabOutDir, ...
            desWidth, desHeight);

        % if there has been a previous step, we can now remove the intermediate
        % results; else, just keep as before and proceed
        if applyPadNCrop 
            rmdir(imgInDir, 's');
            rmdir(labInDir, 's');
        end
        imgInDir = tmpResImgOutDir;
        labInDir = tmpResLabOutDir;
        % inform about where the last step successfully completed
        dispPrint('Resizing completed');
        disp(['Images saved at: ' imgInDir]);
        disp(['Labels saved at: ' labInDir]);
    end
    % color standardization
    if applyColorStd
        dispPrint('Applying color standardization to dataset...');
        if applyColorIdx
            tmpColImgOutDir = fullfile(fileparts(imgOutDir), 'tmpColImgOutDir');
            tmpColLabOutDir = fullfile(fileparts(labOutDir), 'tmpColLabOutDir');
            if exist(tmpColImgOutDir, 'dir') || exist(tmpColLabOutDir, 'dir')
                rmdir(tmpColImgOutDir, 's'); rmdir(tmpColLabOutDir, 's');
            end
            mkdir(tmpColImgOutDir); mkdir(tmpColLabOutDir);
        else
            tmpColImgOutDir = imgOutDir;
            tmpColLabOutDir = labOutDir;
        end

        prepareColorStandardization(imgInDir, tmpColImgOutDir, ...
            labInDir, tmpColLabOutDir, templatesDir);

        if applyPadNCrop || applyResizing
            rmdir(imgInDir, 's');
            rmdir(labInDir, 's');
        end

        imgInDir = tmpColImgOutDir;
        labInDir = tmpColLabOutDir;
        % inform about where the last step successfully completed
        dispPrint('Color Std completed');
        disp(['Images saved at: ' imgInDir]);
        disp(['Labels saved at: ' labInDir]);
    end
    % color indexing
    if applyColorIdx
        dispPrint('Adapting labels to color indexing...');    
        prepareIndexedLabels(imgInDir, imgOutDir, labInDir, labOutDir, ...
            cmap);

        if applyPadNCrop || applyResizing || applyColorStd
            rmdir(imgInDir, 's');
            rmdir(labInDir, 's');
        end
    end
    %----------------------------------------------------- END PROCESSING
    dispPrint(['Final dataset has now ' num2str(numel(dir(imgOutDir))-2) ...
        ' augmented elements']);
    disp(['Images saved at: ' imgOutDir]);
    disp(['Labels saved at: ' labOutDir]);
end