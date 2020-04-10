function [gradientGen, gradientDisc, stateGen] = modelGradients(dlnetGen, dlnetDisc, dlX, dlZ)
% Take the G and D networks with the input data (dlX is a minibatch) and 
% the array of the random values received by the Generator (dlZ)
    
    % Calculate predictions for real data with the discriminator
    dlYPred = forward(dlnetDisc, dlX);
    % Calculate predictions for generated data with the generator
    [dlXGen, stateGen] = forward(dlnetGen, dlZ);
    % Calculate now predictions for generated data with the discriminator
    dlYPredGen = forward(dlnetDisc, dlXGen);
    % Calculate the GAN loss (losses of both players)
    [lossGen, lossDisc] = ganLoss(dlYPred, dlYPredGen);
    % For each net, calculate the gradient basing on the loss. In the case
    % of the generator, the paramenter 'RetainData' indicates if a dlarray
    % retains the derivative trace until the end of dfeval (which evaluates
    % the gradient). Actually this is useful only when the call for
    % evaluation happens more than once: much memory is required
    gradientGen = dlgradient(lossGen, dlnetGen.Learnables, 'RetainData', true);
    gradientDisc = dlgradient(lossDisc, dlnetDisc.Learnables);
          
end