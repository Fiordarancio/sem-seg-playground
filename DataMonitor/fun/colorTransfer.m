function [ norm ] = colorTransfer(imageRGB, targetRGB, verbose)
% Transform a source image w.r.t. a target image according to Reinhard's 
% Color Transfer method.
% Parameters:
% - image   : RGB source image
% - target  : RGB template image
% - verbose : (optional) display results in subplot (default 0)
% 
% References:
% [1] E Reinhard, M Adhikhmin, B Gooch, P Shirley. "Color transfer between 
%     images". IEEE Computer Graphics and Applications, vol.21 no.5, pp.
%     34-41, 2001.
    narginchk(2,3);
    if ~exist('verbose', 'var') || isempty(verbose)
        verbose = 0;
    end
    
    % convert RGB to LAB colour space 
    imageLab = applycform(im2double(imageRGB), makecform('srgb2lab'));
    targetLab = applycform(im2double(targetRGB), makecform('srgb2lab'));
    
    % compute means on each channels
    mi = mean(reshape(imageLab, [], 3));
    mt = mean(reshape(targetLab, [], 3));
    
    % compute standart deviations
    stdi = std(reshape(imageLab, [], 3));
    stdt = std(reshape(targetLab, [], 3));
    
    % normalise each channel based on statistics of source and target images
    norm(:,:,1) = ((imageLab(:,:,1)-mi(1))*(stdt(1)/stdi(1)))+mt(1);
    norm(:,:,2) = ((imageLab(:,:,2)-mi(2))*(stdt(2)/stdi(2)))+mt(2);
    norm(:,:,3) = ((imageLab(:,:,3)-mi(3))*(stdt(3)/stdi(3)))+mt(3);

    % convert again from LAB to RGB 
    norm = applycform(norm, makecform('lab2srgb'));

    % Display results if verbose mode is true
    if verbose
        figure;
        subplot(1,3,1); imshow(imageRGB);   title('Source Image');
        subplot(1,3,2); imshow(targetRGB);   title('Target Image');
        subplot(1,3,3); imshow(norm); title('Normalised (Reinhard)');
    end

end