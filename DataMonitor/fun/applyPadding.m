function newImg = applyPadding(oldImg, padding_width, padding_height, mode)
% Apply a defined amount of horizontal and/or vertical margin filling
% padded pixels with a desired method.
% Inputs:
%   - oldImg:           image to be padded
%   - padding_width:    desired horizontal amount of padding 
%   - padding_height:   desired vertical amount of padding
%   - mode:             method to fill the padded pixels. Possible values:
%                       + 'black':  add black pixels
%                       + 'fill':   padded pixels have the same value of
%                                   the ones encountered on the borders of
%                                   the initial image
%                       + 'mirror': padded pixels repeat backwards the 
%                                   values of the pixels of the original
%                                   image, in the direction of the current
%                                   dimension. Angles are left black
    width = size(oldImg, 2);
    height = size(oldImg, 1);
    if strcmp(mode, 'fill')
        newImg(:, :, 1) = [
            oldImg(:,:,1), oldImg(:,end,1) .* uint8(ones(height, padding_width, 1));
            oldImg(end,:,1) .* uint8(ones(padding_height, width, 1)), zeros(padding_height, padding_width);
            ];
        if size(oldImg, 3) > 1
            newImg(:, :, 2) = [
                oldImg(:,:,2), oldImg(:,end,3) .* uint8(ones(height, padding_width, 1));
                oldImg(end,:,2) .* uint8(ones(padding_height, width, 1)), zeros(padding_height, padding_width);
                ];
            newImg(:, :, 3) = [
                oldImg(:,:,3), oldImg(:,end,3) .* uint8(ones(height, padding_width, 1));
                oldImg(end,:,3) .* uint8(ones(padding_height, width, 1)), zeros(padding_height, padding_width);
                ];
        end
    else
        if strcmp(mode, 'black')
            newImg(:, :, 1) = [
                oldImg(:,:,1), zeros(height, padding_width, 1);
                zeros(padding_height, width, 1), zeros(padding_height, padding_width);
                ];
            if size(oldImg, 3) > 1
                newImg(:, :, 2) = [
                    oldImg(:,:,2), zeros(height, padding_width, 1);
                    zeros(padding_height, width, 1), zeros(padding_height, padding_width);
                    ];
                newImg(:, :, 3) = [
                    oldImg(:,:,3), zeros(height, padding_width, 1);
                    zeros(padding_height, width, 1), zeros(padding_height, padding_width);
                    ];
            end
        else
            if strcmp(mode, 'mirror')
                mirrorWImg = flip(oldImg, 1); % vertical flip
                mirrorHImg = flip(oldImg, 2); % horizontal flip
                newImg(:, :, 1) = [
                    oldImg(:,:,1), mirrorHImg(:, 1:padding_width-1, 1);
                    mirrorWImg(1:padding_height-1, :, 1), zeros(padding_height, padding_width);
                    ];
                if size(oldImg, 3) > 1
                    newImg(:, :, 2) = [
                        oldImg(:,:,2), mirrorHImg(:, 1:padding_width-1, 2);
                        mirrorWImg(1:padding_height-1, :, 1), zeros(padding_height, padding_width);
                        ];
                    newImg(:, :, 3) = [
                        oldImg(:,:,3), mirrorHImg(:, 1:padding_width-1, 3);
                        mirrorWImg(1:padding_height-1, :, 1), zeros(padding_height, padding_width);
                        ];
                end
            else
                disp('Error: recognized modes are\n\r''fill''|''black''|''mirror''');
                newImg = oldImg;
            end
        end
    end
end