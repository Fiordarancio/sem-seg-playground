function [] = dispPrint(message)
    % Just print some -- to divide section in console logging
    for i=1 : length(char(message)) + 1
        fprintf("%s", '-');
    end
    fprintf("\n");
    disp(message);
end