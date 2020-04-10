function [actualResult] = guitarEvaluateResult (net, testImage, labelImage, classes, cmap)
% This function takes an image from the test set and displays results for
% it, both with plot and with statistics
    C = semanticseg(testImage, net); % call for an outoput from our network
    B = labeloverlay(testImage, C, 'Colormap', cmap, 'Transparency', 0.4);
    
    % compare with the ground truth. In this example, if we see a color and
    % not a gray, it means that those areas are different
    % DEBUG NOTE: remember that the cmap should be converted into [0,255]

    actualResult = uint8(C); % conversion
    expectedResult = uint8(labelImage);
    
    figure; 
    tiledlayout(1,2);
    nexttile
    imshow(B); pixelLabelColorbar(cmap, classes); title('Result');
    nexttile
    imshowpair (actualResult, expectedResult); title('False color comparison with label');
    % actual is on green, expected is blue/red channel

    % we should notice that little object tend to be more misunderstood that
    % greater ones (and this does make sense). Let's check statistically using
    % the Jaccard coefficient, which is a similarity coefficient to measure the
    % amount of overlap basing on the intersection-over-union (IoU) metric
    disp('Accuracy indexes:');
    iou = jaccard(C, labelImage)*100; % disp as percentages
    dic = dice(C, labelImage)*100;
    bfs = bfscore(C, labelImage)*100;
    disp(table(classes, iou, dic, bfs));
end