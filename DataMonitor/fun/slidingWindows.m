function [imgs] = slidingWindows(image, stride, winSize)
% Given an image, gives back an array of cropped sub-images of size 
% winSize. The step between each image is stride: a smaller
% stride will require much more computational demand. 
% NOTE: winSize must be [row, columns], so: [height, width]
    imageSize = size(image);
    k = 1;
    for i = 1 : stride(1) : imageSize(1)
        if i+winSize(1)-1 <= size(image,1)
            for j = 1 : stride(2) : size(image,2)
                if j+winSize(2)-1 <= imageSize(2)
                    imgs{k} = image(i:i+winSize(1)-1, j:j+winSize(2)-1, :);
                    imshow(imgs{k});
                    k = k+1;
                end
            end
        end
    end
    % at least 1 image will always be there (worst case: best fit)
    disp (['Extracted ' num2str(k-1) ' sub-image(s).']);
end