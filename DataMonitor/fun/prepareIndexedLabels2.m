function [] = prepareIndexedLabels2(labInDir, labOutDir, indexMap, adjust)
% Converts RGB labels into 1 channel, color indexed labels according to the
% given label map. 
% Parameters.
%   - lab[In/Out]Dir:   input and output folders for labels
%   - indexMap:         usually a pixelLabelIDs cell array, used for 
%                       understanding how each pixel must be interpreted
%   - adjust:           if a label is not recognized and adjust is true, we
%                       try to adjust it putting the value of the most
%                       similar label (not implemented yet)
    
    dispPrint('Convert labels index from:');
    fprintf('\t%s\n', labInDir);
    % create output folders if they do not exist
    clearDir(labOutDir, false);
    % create datastore
    lbds = imageDatastore(labInDir);
    % scroll the 3D matrix: for each triplet in depth, find the index of
    % the corresponding one into the labelIDs map. When found, put it in
    % the new label. If the index is not found, we assume there is pixerror
    % and we put the pixel as <undefined> whose index is assumed to be 0.
    % NOTE: Matlab counts from 1, so every index must be downgraded by 1.
    numClasses = size(indexMap,1);
    pixerror = zeros(length(indexMap),1);
    while hasdata(lbds)
        [L, info] = read(lbds);
        [~, name, ext] = fileparts(info.Filename);
        fprintf('Scanning %s\n', [name ext]);
        [h, w] = size(L, 1:2);
        newL = uint8(zeros(h, w));
        oldPix = L(1,1,:);
        for i=1 : h
            for j=1 : w
                pix = L(i,j,:); % get the pixel
                k = 0;  
                while k < numClasses
                    idx = indexMap{k+1}; % get the label triplet
                    if idx(1) == pix(1) && idx(2) == pix(2) && idx(3) == pix(3)
                        newL(i,j) = k;
                        break;
                    end
                    % save pixerror vector for later use
                    dpix = reshape(pix,[1,3,1]);
                    dpix = double(dpix);
                    pixerror(k+1) = norm(indexMap{k+1} - dpix);
                    k = k+1;
                end
                if k == numClasses
                    if ~isequal(oldPix, pix)
                        fprintf('First at coords %d, %d:\n', i,j);
                        fprintf('No valid index found for [%d, %d, %d]\n', ...
                            pix(1), pix(2), pix(3));
                        oldPix = pix;
                    end
                    % if an unknown label is found, we try to adjust it
                    % with the most similar label. 
                    newL(i,j) = find(pixerror == min(pixerror), 1, 'first');
%                     newL(i,j) = 0;
                end
            end
        end
        imshow(newL);
        imwrite (newL, fullfile(labOutDir, [name ext]));
    end   
end
