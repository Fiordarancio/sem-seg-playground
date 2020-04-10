% Generates Pascal or Motive colormap, depending on the presence of the
% backgound class into it
function [classes, labelIDs, cmap] = motiveClassCmapLabels(hasBackground)
    classes = motiveClasses();
    if ~hasBackground
        classes = classes(2:end);
        cmap = generatePascalCmap(length(classes), false);
    else
        cmap = motiveColorMap;
    end
    
    labelIDs = cell(2,1);
    for i = 1 : length(cmap)
        labelIDs{i} = cmap(i, :);
    end
end