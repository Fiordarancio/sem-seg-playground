function isCorrect = checkImageCorrectness(location, ext, width, height, channels)
% Look onto all the file of 'ext' type under the 'location' directory in
% order to check if all of them have the right size
%     disp('Searching into: '); disp(fullfile(location, ['*.' ext]));
    dinfo = dir(fullfile(location, ['*.' ext]));
    allfiles = fullfile({dinfo.folder}, {dinfo.name});
    if isempty(allfiles)
        disp('Error: no files to copare');
        isCorrect = false;
    else
        isCorrect = true;
        for k = 1 : length(allfiles)
            thisfile = allfiles{k};
            try
                thisinfo = imfinfo(thisfile);
                if ~isfield(thisinfo, 'Width') || ~isfield(thisinfo, 'Height') 
                    fprintf('image information for file "%s" is missing width or height\n', thisfile);
                    isCorrect = false; break;
                end
                if strcmp(ext, 'jpg') && (~isfield(thisinfo, 'NumberOfSamples') || thisinfo.NumberOfSamples ~= 3)
                    fprintf('image information for file "%s" is missing #channels\n', thisfile);
                    isCorrect = false; break;  
                end
                if strcmp(ext, 'png') && (~isfield(thisinfo, 'ColorType') || ~strcmp(thisinfo.ColorType, 'truecolor'))
                    fprintf('image information for file "%s" is missing #channels\n', thisfile);
                    isCorrect = false; break;  
                end
                if thisinfo.Width ~= width
                    fprintf('File "%s" width expected %d, got %d\n', thisfile, width, thisinfo.Width);
                    isCorrect = false; break;
                end
                if thisinfo.Height ~= height
                    fprintf('File "%s" height expected %d, got %d\n', thisfile, height, thisinfo.Height);
                    isCorrect = false; break;
                end
            catch ME
                fprintf('Could not get image information for file:\n\t"%s"\n', thisfile);
                fprintf('Caused by:\n"%s"\n', getReport(ME));
                isCorrect = false; break;
            end
        end
    end
end