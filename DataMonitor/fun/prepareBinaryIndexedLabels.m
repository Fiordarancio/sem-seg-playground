function prepareBinaryIndexedLabels(imgInDir, imgOutDir, labInDir, ...
    labOutDir, cmap)
% Convert RGB labels into 1 channel, color indexed labels of binary classes
    
    createDir(imgOutDir); createDir(labOutDir);
    
    imds = imageDatastore(imgInDir);
    lbds = imageDatastore(labInDir);
    
    % check cmap is in correct format
    if ~isempty(find(cmap < 1 & cmap > 0))
        cmap = uint8(cmap .* 255);
    end
    
    if numel(imds.Files) ~= numel(lbds.Files)
        disp('Error: number of images and labels must match. Please check');
        return;
    end
    
    for idx = 1 : numel(imds.Files)
        [I, Iinfo] = readimage(imds, idx);
        [L, Linfo] = readimage(lbds, idx);
        L = uint8(imbinarize(L));
        imshow(L);
        for i = 1 : size(L, 1)
            for j = 1 : size(L, 2)
                found = false;
                for k = 1 : size(cmap, 1)
                    if size(L,3) == 3
                        if L(i, j, :) == cmap(k, :) 
                            newL(i,j) = k-1;
                            found = true;
                        end
                    else
                        if L(i,j,1) == cmap(k,1)
                            newL(i,j) = k-1;
                            found = true;
                        end
                    end
                end
                if found == false
                    fprintf("L(%d,%d,1) %d, size(L) = (%d,%d,%d)\n", ...
                        i, j, L(i,j,1), size(L,1), size(L,2), size(L,3));
                    newL(i, j) = 255; % IGNORE LABEL VALUE: <undefined>
                end
            end
        end
        % save the image and its label
        newL = uint8(newL);
        [~, name, ext] = fileparts(Iinfo.Filename);
        imwrite (I, fullfile(imgOutDir, [name ext]));
        [~, name, ext] = fileparts(Linfo.Filename);
        imwrite (newL, fullfile(labOutDir, [name ext]));        
        newL = 0;
    end
    
end