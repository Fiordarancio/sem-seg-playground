function [classes, labIDs, cmap] = eventAlphaCLab ()
% Returns utility classes, labels and cmap for reading ONLY instances. No
% relation with original classes are returned, but it can be applied from
% outside to different classes, where proper labels are prepared.
% Alpha channels uses a 10% difference in value for each new instance of a
% certain label. We can store a maximum of 11 instances.
% Example: 
%   Category    Instance    Resulting alpha     Corresponding 255 scale
%   --------    --------    ---------------     -----------------------
%   'person'    1           100                 255
%   'person'    2           90                  230
%   'person'    3           80                  204

    % classes are instances
    classes = flip([        
        "Instance_1"
        "Instance_2"
        "Instance_3"
        "Instance_4"
        "Instance_5"
        "Instance_6"
        "Instance_7"
        "Instance_8"
        "Instance_9"
        "Instance_10"
        "Instance_11"
        ]);
    
    % labelIDs are 255-range of the alpha percentage
    labIDs = flip( (0:1:10) * 10);
    labIDs = round(labIDs * 255 / 100);
    labIDs = table2cell(array2table(labIDs));
    
    % cmap is just rainbow
    cmap = [
        255 0   0
        255 150 0
        255 255 0
        200 255 0
        0   255 0
        0   255 150
        0   255 255
        0   150 255
        0   0   255
        150 0   255
        255 0   255    
        ];
    
%     % White gradients
%     cmap = [flip(0:0.1:1)', flip(0:0.1:1)', flip(0:0.1:1)'] * 255;
    
end