function [] = clearDir(dir)
% If the folder DIR exists, delete ANY included file. Removed items cannot
% be recovered, so use with caution.
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
            answ = input('Add to path? [y/n] ', 's');
            if strcmp(answ, 'y')
                addpath(dir);
            end
            disp('Folder cleared');
        end
        % else you are safe ;)
    else
        mkdir(dir); 
        answ = input('Add to path? [y/n] ', 's');
        if strcmp(answ, 'y')
            addpath(dir);
        end
    end   
end