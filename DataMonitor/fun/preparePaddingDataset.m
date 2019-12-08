function [] = preparePaddingDataset(imgInDir, imgOutDir, labInDir, labOutDir, ...
    desiredWidth, desiredHeight, minWidth, minHeight, ...
    aspectError, percError, swStride)
% Prepare examples of given width x height, basing on PADDING & CROPPING.
% Images and labels are filled with padding in order to math the desired
% aspect ratio; then, several sub-images of the desired size are retrieved
% using sliding windows.
% Parameters:
% - imgInDir, imgOutDir, labInDir, labOutDir: location to read and write
% - desiredWidth, desiredHeight: size of inputs required by the network
% - minWidth, minHeight: tolerance on the size of the image (discard)
% - aspectError: tolerance of error on the aspect ratio given by the size
% - swStride > vector [height,width]: stride for the sliding window
    
    %---------------------------------------------------------------------
    % HARD-CODED DATA
    %---------------------------------------------------------------------
    % clear; close all; clc;
    % imgInDir = 'D:\Ilaria\Guitar_Ilaria\dataset\originalImagesEnhanced';
    % imgOutDir = 'D:\Ilaria\Guitar_Ilaria\dataset\imagesResizedDeeplab';
    % labInDir = 'D:\Ilaria\Guitar_Ilaria\dataset\originalLabelEnhanced';
    % labOutDir = 'D:\Ilaria\Guitar_Ilaria\dataset\labelsResizedDeeplab';
    
    % minWidth = 400;
    % minHeight = 300;
    %---------------------------------------------------------------------

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

    % while there are data in the datastore, read an Image and a Label,
    % padding both of them and cropping. Each crop will be saved in the
    % indicated folder
    k = 1;
    disp('Scanning dataset for pad&crop...');
    while hasdata(imds)
        fprintf("\n");
        
        [I, Iinfo] = read(imds); 
        [L, Linfo] = read(lbds);
        % remember that width and height are inverted because matlab reads 
        % the image as it was a matrix!
%         iminfo = imshow(I);
%         disp(info); disp(iminfo); 
        width = size(I,2);
        height = size(I,1);
        aspectRatio = width / height;
        fprintf("Image size %d x %d - Aspect ratio %.3f\n", width, height, aspectRatio);
        
        baseRatio = desiredWidth / desiredHeight;
        %------------------------------------------------- RGB HARDCODED!
        basePSize = desiredWidth * desiredHeight * 3;
        %------------------------------------------------- RGB HARDCODED!
        
        % Pre-check 1: we assume Image and Label are exactly equal in size,
        % so that we can apply the same operations on both, seamlessly
        if width ~= size(L, 2) || height ~= size(L, 1)
            disp('Error: sizes of image and label don''t correspond. Please check');
            continue; % just discard
        end

        % Pre-check 2: images must be RGB
        if size(I,3) == 1
            disp('Error: images must be RGB. Please check');
        end

        % Check number 1: the image must be greater or equal a minimum size
        if width <= minWidth && height <= minHeight
            % drop the image, don't save it
            disp('This image is too small and will be discarded from the dataset');
            continue;
        else
            % Check number 2: is the image definitively "smaller" than our
            % requested size? Which means, both dimensions are smaller than the
            % one requested by the network?
            if width <= desiredWidth && height <= desiredHeight
                disp('Smaller image: requires padding');
                padding_height = desiredHeight - height;
                padding_width = desiredWidth - width;
            else
                % Check number 3: the image is bigger or without a proper
                % aspect ratio for our network. Padding is needed before
                % cropping the input image, preserving indeed aspect ratios
                if aspectRatio < (baseRatio-aspectError) || aspectRatio > (baseRatio+aspectError)
                    disp('Aspect ratio is not proper: padding is needed');
                    if aspectRatio > baseRatio || aspectRatio <= 1.00 
                        if width > height
                            padding_height = round( (width * desiredHeight) / desiredWidth ) - height;
                            padding_width = 0;
                        else
                            % in straight rects and squares, enlarge the horizontal
                            padding_width = round( (height * desiredWidth) / desiredHeight ) - width;
                            padding_height = 0;
                        end
                    else % we should do the contrary
                        if width > height
                            % in straight rects and squares, enlarge the horizontal
                            padding_width = round( (height * desiredWidth) / desiredHeight ) - width;
                            padding_height = 0;
                        else
                            padding_height = round( (width * desiredHeight) / desiredWidth ) - height;
                            padding_width = 0;
                        end
                    end
                else
                    % If we arrive at this point, and no padding is needed,
                    % then we have an image with a correct aspect ratio (4:3)
                    % with dimensions greater or equal to 960x720
                    disp('This image has good aspect ratio');
                end
            end
            disp(['Required paddings [' num2str(padding_width) ' ' num2str(padding_height) ']']);
            newI = applyPadding(I, padding_width, padding_height, 'black');
            newL = applyPadding(L, padding_width, padding_height, 'black');
%             newL = uint8(imbinarize(newL));

            newWidth = size(newI,2);
            newHeight = size(newI,1);
            
            fprintf("New image/label size %d x %d - Aspect ratio %.3f\n", ...
                newWidth, newHeight, newWidth/newHeight);

            % Step number 4: at this point, our image is clearly greater or
            % equal than 960x720, as required by DeepLab. Hence we need to crop
            % sub-images (with relative labels), using here a sliding window
            % fashion
%             swStride = gcd(960, 720) * [1/10 1/10] % number of pixels to jump 
            % choose the stride as a divisor of the size of the image in order
            % to let it fit correctly the image
            imgs = slidingWindows(newI, swStride, [desiredHeight, desiredWidth]);
            labs = slidingWindows(newL, swStride, [desiredHeight, desiredWidth]);
            
            % Step number 5: save all the new images + labels into the new
            % folder, numbering them by new as they were all different
            for i = 1 : length(imgs)
                % discard images which have no labels (or too few part)
                nonzero = find(labs{i});
                fprintf("Label percentage: %.1f%%\n", 100*length(nonzero)/basePSize);
                if ~isempty(nonzero) && length(nonzero)/basePSize > percError
                    imshow(newI); 
                    imshow(newL);
                    imwrite(imgs{i}, fullfile(imgOutDir, [num2str(k) '.jpg']) );
                    imwrite(labs{i}, fullfile(labOutDir, [num2str(k) '.png']) );
                    k = k+1;
                else
                    disp('Crop discarded');
                end
            end
        end
    end
    disp ('New padding&crop - based dataset created');
end
