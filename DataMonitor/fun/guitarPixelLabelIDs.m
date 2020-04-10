function labelIDs = guitarPixelLabelIDs(isCategorical)
% Return a cell array with label IDs for each class. 
%   - isCategorical: if true, labels are N categorical indexes into [0, N)
%                    if false, labels are RGB triplets
    if ~isCategorical
        labelIDs = { ...
            % "Background"
            [
            0 0 0; ...
            ]
            % "Guitar
            [
            1 1 1; ... 
            ]
        };
    else
        % NOTE: labels are gategorically saved as 0,1. It does not like to use,
        % for guitar, 255 255 255; ... 
        labelIDs = { ...
            0, ... % "Background"
            1, ... % "ElectricGuitar_Bass"
%             2, ... % "Classicguitar"
%             3, ... % "Drums"
%             4, ... % "MicrophoneStand"
%             5, ... % "Person"
%             6, ... % "Keyboard"
            };
    end
    
end