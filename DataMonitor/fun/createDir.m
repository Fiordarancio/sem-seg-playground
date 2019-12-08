function [] = createDir(dir)
% If DIR does not exist, creates it
    if ~ischar(dir) && ~isstring(dir) && iscellstr(dir)
        error([mfilename ': Invalid input dir'], ...
            'dir must be a string with a valid path');
    end
    if ~exist(dir, 'dir')
        mkdir(dir);
    end        
end