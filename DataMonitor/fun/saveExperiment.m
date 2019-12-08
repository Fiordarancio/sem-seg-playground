% function [] = saveExperiment(outDir, netName, saveDs)
%     % THIS FUNCTION CAN BE DANGEROUS. IT DOES NOT PASS THE WORKSPACE WE ARE
%     % ACTUALLY WORKING ON.
%     
%     % Utility for saving the known variables into a workspace. Variables
%     % are related to Segnet, Deeplab and Unet experiments. 
%     % Inputs:
%     %   - outDir:   path where the files must be saved
%     %   - netName:  name of the network type (Segnet|Unet|Deeplab...)
%     %   - saveDs:   if true, save Datastpres in a parallel file
%     dateTime = datetime();
%     dateTime.Format = 'yyyy-MM-dd_HH-mm-ss';
%     filename = [char(netName) '-' char(dateTime) '-workspace.mat'];
%     filename = fullfile(outDir, filename);
%     dispPrint('Saving experiment workspace at: ');
%     disp(filename);
%    
%     %----------------------------------------------------- BEGIN USER CODE
%     % define the variables you want to save. Note that MATLAB can encounter
%     % errors while loading Datastores, so it is better to save only
%     % information that are relevant to construct them
%     save(filename, ...
%         ... % dataset objects
%         'augmenter', ...% the used augmenter
%         'classes', ...  % the used classes (names)
%         'cmap', ...     % the used color map
%         'frequency', ...% evaluated frequency of labels in the dataset
%         'imageSize', ...% input dimension [h, w, ch] of examples
%         'labelIDs', ... % the used labelsIDs
%         'lgraph', ...   % the used lgraph (yet not constructed)
%         'trainperc', ...% percentage of dataset used for training
%         'valperc', ...  % percentage of dataset used for 
%         ... % training options
%         'doTraining', ...   % if results have been obtained by training
%         'options', ...  % training options (contains everything relevant)
%         'net', ...      % the trained natwork, USABLE FOR FUTURE INFERENCE
%         ... % paths
%         'datasetDir', ...   % path to main dataset's folder
%         'guitarCpDir', ...  % path to checkpoints folder
%         'guitarDir', ...    % path to main folder
%         'guitarResDir', ... % path to results folder
%         'imgDir', ...       % path to images set
%         'labDir', ...       % path to labels set
%         'pretrainedFolder', ... % path to original Matlab pretrained net
%         'testLabelDir' ...  % path to save test labels
%         );
%     %------------------------------------------------------- END USER CODE    
%     for i=1 : length(variables)
%         save(filename, char(variables(i)));
%     end
%         
%     if saveDs
%         filename = [char(netName) '-' char(dateTime) '-datastore.mat'];
%         save(filename, ...
%             'imds', ...     % main imageDatastore
%             'imdsTest', ... % test split
%             'imdsTrain', ...% train split
%             'imdsVal', ...  % val split
%             'pxds', ...     % main pixelLabelDatastore
%             'pxdsTest', ... % test split  
%             'pxdsTrain', ...% train split
%             'pxdsVal', ...  % val split
%             'pximds', ...   % examples imdsTrain + pxdsTrain using augmenter
%             'pximdsVal' ... % examples imdsVal + pxdsVal
%             );
%         dispPrint('Saving experiment datastores at: ');
%         disp(filename);
%         save(filename, variables);
%     end
% end