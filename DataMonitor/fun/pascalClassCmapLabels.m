function [classes, labelIDs, cmap] = pascalClassCmapLabels()
    classes = [
        "background"
        "aeroplane"
        "bicycle"
        "bird"
        "boat"
        "bottle"
        "bus"
        "car"
        "cat"
        "chair"
        "cow"
        "diningtable"
        "dog"
        "horse"
        "motorbike"
        "person"
        "pottedplant"
        "sheep"
        "sofa"
        "train"
        "tvmonitor"
%         "void"
    ];
    cmap = generatePascalCmap(21, false);
    labelIDs = cell(2,1);
    for i = 1 : length(cmap)
        labelIDs{i} = cmap(i, :);
    end
end