function [newImage] = paddingToAspectRatio(image, desiredWidth, ...
                                            desiredHeight, varargin)
% Given a generic image, returns a new one of size [desiredWidth, 
% desiredHeight] and desired aspect ratio (that is (WIDTH / HEIGHT), using 
% padding. No resize is applied.
% Inputs:
%   - image:                generic M-by-N-by-C image, where C could be any
%                           number of channels the image has
%   - desiredWidth:         desired width (for matrices, it is num cols)
%   - desiredheight:        desired height (for matrices, it is num rows)
%   - tolerance:            range of tolerance with the given aspect ratio
%                           (default: 0)
%   - padMode:              how to fill the padded pixels. Allowed values
%                           are 'black'|'mirror'|'fill'. Check applyPadding
%                           for more information.
% Output:
%   - newImage:             image with black padding applied
    
    % validate arguments
    narginchk(3,5);
    if isempty(varargin)
            tolerance = 0;
            padMode = 'black';
    else
        if length(varargin) == 1
            tolerance = str2double(varargin{1});
            padMode = 'black';
        else
            tolerance = str2double(varargin{1});
            padMode = varargin{2};
        end
    end
    
    desiredAspectRatio = desiredWidth / desiredHeight;
    dispPrint(['Desired image with aspect ratio: ' ...
        num2str(desiredAspectRatio)]);
    disp(['Desired range [' num2str(desiredAspectRatio-tolerance) ...
        ', ' num2str(desiredAspectRatio+tolerance) ']']);
    
    % get the size of the image
    [height, width, ~] = size(image);
    % it must be smaller to apply padding
    if height > desiredHeight || width > desiredWidth
        disp('Error: image is bigger than desired sizes. Crop not allowed');
        newImage = 0;
        return;
    end
    % actual aspect ratio
    aspectRatio = width / height;
    fprintf("Image size %d x %d - Aspect ratio %.3f\n", width, height, aspectRatio);
    
    if aspectRatio < (desiredAspectRatio-tolerance) || ...
       aspectRatio > (desiredAspectRatio+tolerance)
        disp('Aspect ratio is not proper: padding is needed');
        if aspectRatio > desiredAspectRatio || aspectRatio <= 1.00 
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
        disp(['Required paddings [' num2str(padding_width) ' ' ...
            num2str(padding_height) ']']);
    else
        % If we arrive at this point, and no padding is needed, then we 
        % have an image with a correct aspect ratio 
        disp('This image has good aspect ratio');
        padding_width = 0;
        padding_height = 0;
    end
    
    newImage = applyPadding(image, padding_width, padding_height, padMode);

    newWidth = size(newImage,2);
    newHeight = size(newImage,1);

    fprintf("New image/label size %d x %d - Aspect ratio %.3f\n", ...
        newWidth, newHeight, newWidth/newHeight);
end