function [] = clearDir(dir, addPath)
% If the folder DIR exists, delete ANY included file. Removed items cannot
% be recovered, so use with caution.
    narginchk(2,3);
    if ~ischar(dir) && ~isstring(dir) 
        error([mfilename ': Invalid input dir'], ...
            'dir must be a string with a valid path');
    end
    if exist(dir, 'dir')
        fprintf('Attempting to remove:\n\t%s\n', dir);
        answ = input('Are you sure? [y/n] ', 's');
        if strcmp(answ, 'y')             
            rmdir(dir, 's');
            mkdir(dir); 
            if addPath
                addpath(dir);
            end
            disp('Folder cleared');
        end
        % else you are safe ;)
    else
        fprintf('Creating new dir:\n\t%s\n', dir);
        mkdir(dir); 
        if addPath
            addpath(dir);
        end
    end   
end