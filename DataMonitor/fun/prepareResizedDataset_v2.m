function [] = prepareResizedDataset_v2(imgInDir, imgOutDir, labInDir, labOutDir, ...
    desiredWidth, desiredHeight, resizeBigger);
% Apply resizing on given images preserving the aspect ratio. Images are
% resized to reach the maximum size smaller or equal the desired one, so
% that the initial aspect ratio is preserved.
% Inputs:
%   - img[In/Out]Dir, lab[In/Out]Dir:   input and output folders;
%   - desired[Width/Height]:            desired sizes
%   - resizeBigger:                     if true, images with size bigger
%                                       than the desired one will be
%                                       downscaled (default: false)

    if resizeBigger ~= false || resizeBigger ~= true
        resizeBigger = false;
    end
    
    createDir(imgOutDir); createDir(labOutDir);

    imds = imageDatastore(imgInDir);
    lbds = imageDatastore(labInDir);
    % check number of elements: must correspond
    if numel(imds.Files) ~= numel(lbds.Files)
        disp('Error: number of images and labels must match. Please check');
        return;
    end

    % scroll
    dispPrint('Scanning dataset for resizing...');
    while hasdata(lbds)
        % read the following item from a datastore. Subsequent calls
        % continue reading from the endpoint of the previous call.
        [I, Iinfo] = read(imds);
        [L, Linfo] = read(lbds);
        [height, width, ~] = size(I);
        % if the image is bigger, resize only if commanded, else resize
        % accordingly to original aspect ratio.
        if height > desiredHeight || width > desiredWidth
            if resizeBigger
                if width >= height
                    dw = width - desiredWidth
                    scale = 1 - dw/width
                else
                    dh = height - desiredHeight
                    scale = 1 - dh/height
                end
                I = imresize(I, scale);
                L = imresize(L, scale);
                fprintf('Downsized to %d x %d.\n', size(I,2), size(I,1));
            else
                disp('No downsizing applied.');
            end
        else
            if width >= height
                dw = desiredWidth - width;
                scale = 1 + dw/width;
            else
                dh = desiredHeight - height;
                scale = 1 + dh/height;
            end
            I = imresize(I, scale);
            L = imresize(L, scale);
            fprintf('Upsized to %d x %d.\n', size(I,2), size(I,1));
        end
        
        % write to disk
        [~, filename, ext] = fileparts(Iinfo.Filename);
        imwrite(I, fullfile(imgOutDir, [filename ext]) ); % the fast strcat
        [~, filename, ext] = fileparts(Linfo.Filename);
        imwrite(L, fullfile(labOutDir, [filename ext]) );
    end
    dispPrint('Aspect-ratio-aware resizing completed');
end