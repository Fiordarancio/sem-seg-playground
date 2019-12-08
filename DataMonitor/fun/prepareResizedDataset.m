function [] = prepareResizedDataset(imgInDir, imgOutDir, labInDir, labOutDir, ...
    desiredWidth, desiredHeight);
% Brute force resizing of images, preserving RGB channel

    if ~exist(imgOutDir, 'dir')
        mkdir(imgOutDir);
    end
    if ~exist(labOutDir, 'dir')
        mkdir(labOutDir);
    end

    imds = imageDatastore(imgInDir);
    lbds = imageDatastore(labInDir);
    % check number of elements: must correspond
    if numel(imds.Files) ~= numel(lbds.Files)
        disp('Error: number of images and labels must match. Please check');
        return;
    end

    disp('Scanning dataset for resizing...');
    % scroll over the entire folder
%     imIdx = 1; % use to ignore already prepared images
%     while imIdx < numel(imds.Files)
    while hasdata(lbds)
        % read the following item from a datastore. Subsequent calls continue 
        % reading from the endpoint of the previous call.
        [I,info] = read(imds);
%         [I,info] = readimage(imds, imIdx);
        % resize image
        I = imresize(I, newSize);  
        % write to disk
        [~, filename, ext] = fileparts(info.Filename);
        imwrite(I, fullfile(imgOutDir, [filename ext]) ); % the fast strcat
%         subplot(1,2,1); 
        imshow(I);

        [L, info] = read(lbds);
        L = uint8(imresize(L, newSize));
        % remove colors in the middle by assigning only [0,1] values
        L = uint8(imbinarize(L)); 
        % write to disk
        [~, filename, ext] = fileparts(info.Filename);
        imwrite(L, fullfile(labOutDir, [filename ext]) );
%         subplot(1,2,2);
        imshow(L);

%         imIdx = imIdx+1;
    end
    disp('Dataset for DeepLab training successfully created');

end