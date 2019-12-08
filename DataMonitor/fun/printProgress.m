function [] = printProgress (percentage, frequency)
% simple utility to print down a progress 'percentage' every 'frequency'
% checkpoints
    persistent checkpoints;
    persistent step;
    
    if percentage < 0.0 || percentage > 1.0
        disp('Error: percentage must be in [0,1]');
        return;
    end
    
    if percentage == 0
        checkpoints = ones(frequency, 1);
        step = 1 / frequency;
    else
        last_checkpoint = find(checkpoints, 1, 'first');
        if percentage >= (last_checkpoint * step)
            fprintf("%d%% processed\n", uint8(percentage * 100));
            checkpoints(last_checkpoint) = 0;
        end
    end
end