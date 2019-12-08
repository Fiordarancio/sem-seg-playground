% just check again channels in images
datasetDir = '/home/ilaria/Guitar_Ilaria/dataset';

% mainDir = fullfile(datasetDir, 'imagesSegnet');
mainDir = datasetDir;
subdirs = {
%     'drum'
%     'flute'
%     'guitar'
%     'violin'
    'imagesSegnet'
    'labelsSegnet'
    };

parfor i = 1 : length(subdirs)
    subDir = fullfile(mainDir, subdirs{i});
    imds = imageDatastore(subDir);
    count = 0;
    while hasdata(imds)
        [I, info] = read(imds);
        if size(I,3) ~= 3
            count = count +1;
            I(:,:,2) = I(:,:,1);
            I(:,:,3) = I(:,:,1);
            [path, name, ext] = fileparts(info.Filename);
            imwrite(I, info.Filename);
        end
    end
    disp(['Found ' num2str(count) ' non RGB images'])
end