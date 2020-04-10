function pxds_cat = guitarCategoricalLabels(pxds, cmap, dir)
% Convert images in pixelDatastore to categorical string labels
% Converts a label for segmentation which is described by
% RGB triplets to another label described by categorical values (scalars).
% That is, for each color a category is assigned.
% The function takes as input:
% - an LabelImageDatastore in which a set of strings containing the files 
%   with the RGB-based labels are saved
% - the color map from which categories can be extracted
% - the folder in with the new labels are supposed to stay
% The function outputs the same datastore which now refers to an array of 
% strings containing the paths of the correspondent category-based files

    numFiles = length(pxds_rgb.PixelLabelData);
    location = fullfile(dir, 'categorical_labels\');
    if ~exist(location,'dir')
        disp('Creating categorical labels in: ');
        disp(location);
        mkdir(location);
    end
    for i=1 : numFiles
        oldlabel = imread(pxds_rgb.PixelLabelData{i});
        newlabel = rgb2ind(oldlabel, cmap);
        % save the newlabel
        imwrite(newlabel, fullfile(location, strcat('cat_', num2str(i), '.png')));
    end
    % create new pixelLabelDatastore in order assigning the new location
    % and keeping the other values the same as before
    pxds_cat = pixelLabelDatastore(location, pxds_rgb.ClassNames, camvidPixelLabelIDs())
    
end