% Visualize, analyze and prepare the examples that contain all the classes
% of EventDataset. Notice that both class and instances are visualized.
% This is an utility program, divided into sections. You can use it in 
% order to monitor if a defined bunch of provided labels is correct or not
% for further use
%--------------------------------------------------------------------------
clear; close all; clc;
%-------------------------------------------------------- BEGIN USER CODE
dataPath = 'C:\Users\Ilaria\Guitar_Ilaria\dataset';
eventPath = fullfile(dataPath, 'EventDataset');
%---------------------------------------------------------- END USER CODE
imgInDir = fullfile(eventPath, 'eventImages');
labInDir = fullfile(eventPath, 'eventLabels');
labOutDir = fullfile(eventPath, 'rgbLabels'); % TEMPORARY, for visualization
alfOutDir = fullfile(eventPath, 'alphaLabels');
clearDir(labOutDir); clearDir(alfOutDir);

%% prepare datastores
dispPrint('Preparing datastores...');
imds = imageDatastore(imgInDir, 'FileExtensions', {'.jpg', '.png'});
lbds = imageDatastore(labInDir, 'FileExtensions', '.png');

if numel(imds.Files) ~= numel(lbds.Files)
    rmdir(imgInDir,'s'); mkdir(imgInDir);
    % pick up from the concert dataset
    imgDir1 = fullfile(dataPath, 'ConcertDataset', 'images');
    imgDir2 = fullfile(imgDir1, 'not_yet_labeled');
    for i=1 : numel(lbds.Files)
        [~, name, ~] = fileparts(lbds.Files{i});
%         imname = fullfile(imgDir1, [name '.jpg']);
%         if exist(imname, 'file')
%             copyfile(imname, imgInDir);
%         else
%             imname = fullfile(imgDir1, [name '.png']);
%             if exist(imname, 'file')
%                 copyfile(imname, imgInDir);
%             else
                imname = fullfile(imgDir2, [name '.jpg']);
                if exist(imname, 'file')
                    copyfile(imname, imgInDir);
                else
                    imname = fullfile(imgDir2, [name '.png']);
                    if exist(imname, 'file')
                        copyfile(imname, imgInDir);
                    else
                        error(['Error: file ' imname ...
                            ' not found in original dataset']);
                    end
                end
%             end
%         end
    end
end
imds = imageDatastore(imgInDir, 'FileExtensions', {'.jpg', '.png'});

%% Start analysis
dispPrint('Analyzing images features...');
classes = eventClasses
labids = eventPixelLabelIDs(false);
cmap = eventColorMap;

% % the function readimage cannot read the alpha channel, so we need to use
% % another way. However, for visualization, it blends the alpha channel with
% % the RGB so that the instances actually look different (the color changes
% % in brightness and saturation)
% idx = 1;
% I = readimage(imds, idx);
% subplot(2,2,1); imshow(I); title('Original image');
% L = readimage(lbds, idx);
% subplot(2,2,2); imshow(L); title('readimage(datastore): alpha is blended');
% [L, ~, alpha] = imread(lbds.Files{idx});
% subplot(2,2,3); imshow(L); %image(L, 'AlphaData', alpha); 
% title('imread(file): rgb only channels');
% subplot(2,2,4); imshow(alpha); title('imread(file): alpha channel');

% our images use the alpha channel to define instances. In this case, we
% cannot use pixelLabelDatastore directly, because it will interpret each
% quartet RGBA as a different labelID. We then need to save all the RGB
% only labels to retrieve the classes only
for idx = 1 : numel(lbds.Files)
%     % drawing utility 1 (uncomment to visualize)
%     I = readimage(imds, idx);
%     subplot(2,2,1); imshow(I); title('Original image');
%     xlabel(imds.Files{idx});
%     L = readimage(lbds, idx);
%     subplot(2,2,2); imshow(L); title('readimage(datastore): alpha is blended');
    [L, ~, alpha] = imread(lbds.Files{idx});
    if isempty(alpha)
        fprintf('Apparently, all instances have index 1 in:\n\t%s\n', ...
            lbds.Files{idx});
        alpha = uint8(ones(size(L,1:2))*255);
    end
    % update the Label with the aplha 0 as automatically UNDEFINED, in
    % order to avoid errors with correct pixel label indexing etc. In this
    % way, this acts automatically as IGNORE_LABEL
    ignorables = find(alpha == 0);
    L1 = L(:,:,1); L2 = L(:,:,2); L3 = L(:,:,3);
    L1(ignorables) = 0; L2(ignorables) = 0; L3(ignorables) = 0;
    L = cat(3, L1,L2,L3);
%     % drawing utility 2 (uncomment to visualize)    
%     subplot(2,2,3); imshow(L); %image(L, 'AlphaData', alpha); 
%     title('imread(file): rgb only channels');
%     subplot(2,2,4); imshow(alpha); title('imread(file): alpha channel');
    
    [~, lname, lext] = fileparts(lbds.Files{idx});
    imwrite(L, fullfile(labOutDir, [lname lext])); % rgb
    imwrite(alpha, fullfile(alfOutDir, [lname lext])); % alpha
end

%% read new labels, alphas and pixel labels from the new dirs
dispPrint('Reading new labels...');
nwds = imageDatastore(labOutDir, 'FileExtensions', {'.jpg', '.png'});
alds = imageDatastore(alfOutDir, 'FileExtensions', {'.png'});
if numel(nwds.Files) ~= numel(imds.Files)
    error('Error in previous code: numbers of images and labels must match.');
end
pxds = pixelLabelDatastore(labOutDir, classes, labids);
[aclasses, alabids, acmap] = eventAlphaCLab;
apxds = pixelLabelDatastore(alfOutDir, aclasses, alabids);
% count pixel label occurrences
labTbl = countEachLabel(pxds);
disp (labTbl);
frequency = labTbl.PixelCount/sum(labTbl.PixelCount);

%% iterate over
for idx = 1 : numel(lbds.Files)
    % DEBUG NOTE: readimage cannot read the alpha channel, so we need to use
    % another way. However, for visualization, it blends the alpha channel with
    % the RGB so that the instances actually look different (the color changes
    % in brightness and saturation)
    [L, info] = readimage(lbds, idx);
    
    % Using tight_subplot. For more info, type:
    %   >> help tight_subplot
    figure('WindowState', 'maximized', 'Name', info.Filename);
    [handleAxes, ~] = tight_subplot( 2, 3, ...      % rows, columns
                                    [.04 .03], ...  % gap between axes
                                    [.15 .05], ...  % vertical margin
                                    [.02 .01]);     % horizontal " "
    % 1. plot blended label (axes is needed to select the position)
    axes(handleAxes(1)); imshow(L); 
    title('Read from datastore: alpha is blended');
    % 2. plot RGB label
    L = imread(nwds.Files{idx});
    axes(handleAxes(2)); imshow(L);
    title('Read with imread: alpha is retrieved as different matrix');
    % 3. plot ALPHA channel of previous label
    alpha = imread(alds.Files{idx});
    axes(handleAxes(3)); imshow(alpha); title('Alpha channel');
    % 4. plot the frequency graph
    axes(handleAxes(4));
    bar(1:numel(classes), frequency); % numel: number of array elements
    title('Frequency of classes over all dataset');
    xticks(1:numel(classes)); xticklabels(labTbl.Name); xtickangle(45);
    ylabel('Frequency');
    % 5. plot image with overlaid categories
    I = readimage(imds, idx);
    P = readimage(pxds, idx);
    axes(handleAxes(5));
    imshow(labeloverlay(I, P, 'Colormap', cmap/255, 'Transparency', 0.35));
    title('Image with labels overlaid');
    pixelLabelColorbar(cmap/255, classes);

    % 6. plot instances: color varies independently from categories, that 
    % is, object with same instance have different colors.
    A = readimage(apxds, idx);
    axes(handleAxes(6));
    imshow(labeloverlay(I, A, 'Colormap', acmap/255, 'Transparency', 0.5));
    title('Map of instances: object of same instance have same color');
    pixelLabelColorbar(acmap/255, aclasses);

end