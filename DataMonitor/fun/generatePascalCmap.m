function [cmap] = generatePascalCmap(num_classes, normalize)
    narginchk(2,2);
    cmap = zeros(num_classes, 3);
    for i=1:num_classes
        r = uint8(0); g = uint8(0); b = uint8(0);
        c = uint8(i);
        for j=1:8
            r = bitor(r, bitshift(bitget(c, 1), uint8(8-j)));
            g = bitor(g, bitshift(bitget(c, 2), uint8(8-j)));
            b = bitor(b, bitshift(bitget(c, 3), uint8(8-j)));
            c = bitsra(c, 3);
        end
        cmap(i, :) = [r, g, b];
    end
    if normalize
        cmap = float(cmap) / 255;
    end
end
