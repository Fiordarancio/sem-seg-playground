function [lossGen, lossDisc] = ganLoss(dlYPred, dlYPredGen)
% Given the final output of the dispriminator (which is the output of the
% GAN too), calculate loss functions both for D and G.
    
    % Calculate losses for discriminator
    lossReal = -mean(log(sigmoid(dlYPred)));
    lossGenerated = -mean(log(1-sigmoid(dlYPredGen)));
    lossDisc = lossReal + lossGenerated;
    
    % Calculate loss for the generator
    lossGen = -mean(log(sigmoid(dlYPredGen)));
end