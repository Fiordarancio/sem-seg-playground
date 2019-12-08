function cmap = guitarColorMap()
% Define the colormap used by guitar dataset. Values of RGB are in [0,1]
    cmap = [
        0 0 0       % Background
%         255 255 255 % Guitar
        0 150 225    % Guitar (if you prefer to see a color)
        ];

    % Normalize between [0 1]: used for imshow, be careful
    cmap = cmap ./ 255;
end
