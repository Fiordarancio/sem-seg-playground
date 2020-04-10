function cmap = openImageColorMap_guitar()
    % for creating randomly, use the following
%     [~, c] = openImageClasses();
%     cmap = rand(length(c), 3, 'single', 'distributed');

    cmap = [ 
        0 0 0
        255 255 255
    ];

    % si necesitas
%     cmap = uint8(cmap .* 255);
end