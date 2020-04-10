function classTable = openImageClasses()
% this function returns a table containing label ID in the first column and
% the label human-readable name on the right
    path = fullfile(pwd,'class-descriptions-boxable.csv');
    classTable = readtable(path);
end