function numeric_matrix = ctg2num (categorical_matrix, map)
% Converts a matrix from categorical to numeric values, using a given
% map of classes. THIS IS NOT STANDARD NOR OPTIMIZED YET
    for i=1 : size(categorical_matrix, 1)
        for j=1 : size(categorical_matrix, 2)
            catidx = find(map == string(categorical_matrix(1,1)));
            if ~isempty(catidx)
                numeric_matrix(i,j) = catidx - 1;
            else
                numeric_matrix(i,j) = numel(map) - 1;
            end
        end
    end
end