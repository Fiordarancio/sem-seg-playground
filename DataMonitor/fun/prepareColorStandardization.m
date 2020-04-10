function [] = prepareColorStandardization( imgInDir, imgOutDir, ...
    labInDir, labOutDir, templateImgsDir )
% Transform the color map of an image (or a set of images) using another
% template image, saving it in a new location.
% Parameters:
% - imgInDir, labInDir: folder of the images to be transformed
% - templateImgsDir: folder containing the images used as 'color template'
% - imgOutDir, labOutDir: destination folder for the generated examples

    %----------------------------------------------------------------------
    % HARD-CODED DATA
    %----------------------------------------------------------------------
    % image_augmentation('C:\projekt_esp\all\V2\','C:\projekt_esp\all\V2\augmentation\',3, {'c1', 'c2', 'c3'}, 227  )
    % home_path=path to directory with data
    % new_data_path= directory for new data
    % masks_extension = 'jpg', 'png' ...
    % nb_of_class - the number of different groups/folders
    % category - name opf class/category 
    % im_size - size of new images 
    % category = {'c1', 'c2', 'c3' ,'c4'};
    %----------------------------------------------------------------------
    
    if ~exist(imgOutDir, 'dir')
        mkdir(imgOutDir);
    end
    
    if ~exist(labOutDir, 'dir')
        mkdir(labOutDir);
    end

    % create datastores with source images and template images
    sourceImds = imageDatastore (imgInDir);
    sourceLbds = imageDatastore (labInDir);
    templateDs = imageDatastore (templateImgsDir);
   
    % print out the number of templates
    fprintf("Applying %d templates on %d images...\n", ...
        numel(templateDs.Files), numel(sourceImds.Files));
    
    % scan the source datastore. For each image, if they have proper type,
    % execute a color normalization using Reinhard, then save the new image
    % and the relative label. 
    % NOTE: the labels must correspond exactly in name to the ones read by
    % the source image datastore. In this way, we can read the datastore
    % using iterators (hasdata()) and keep track of the progress. On the
    % contrary, the templates' datastore must be scanned in loop
    
    k = 1; % new image index for saving
    disp('Scanning dataset for color standardization...');
    printProgress(0, 100);
    while hasdata(sourceImds)
        [I, Iinfo] = read(sourceImds);
        [L, Linfo] = read(sourceLbds);
        
        [~, ~, imext] = fileparts(Iinfo.Filename);
        [~, ~, lbext] = fileparts(Linfo.Filename);
        
        % check format
        if ~(strcmp(imext, '.jpg') || strcmp(imext, '.png') || ...
                strcmp(imext, '.tif') || strcmp(imext, '.png'))
            fprintf("Error: %s is not a recognized format\n", imext);
            continue;
        end
        if ~(strcmp(lbext, '.jpg') || strcmp(lbext, '.png') || ...
                strcmp(lbext, '.tif') || strcmp(lbext, '.png'))
            fprintf("Error: %s is not a recognized format\n", lbext);
            continue;
        end
        
        % scan the templates
        for i = 1 : numel(templateDs.Files)
           T = readimage(templateDs, i);
           try
               N = NormReinhard(I, T, 0);
               imwrite (N, fullfile(imgOutDir, [num2str(k) imext]));
               imwrite (L, fullfile(labOutDir, [num2str(k) lbext])); % label is just the same
               k = k+1;
           catch exception
               fprintf("%s\n", getReport(exception));
               throw(exception);
           end
        end
        % save also a copy of the original one
        imwrite (I, fullfile(imgOutDir, [num2str(k) imext]));
        imwrite (L, fullfile(labOutDir, [num2str(k) lbext]));
        k = k+1;
        
        printProgress(progress(sourceImds), 100);
    end
    