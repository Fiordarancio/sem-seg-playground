classdef SigmoidLayer < nnet.layer.Layer
    % SigmoidLayer   Custom class for handling sigmoid-based 
    %                activation layer.
    %
    %   SigmoidLayer properties:
    %       Name                   - A name for the layer.
    %       NumInputs              - The number of inputs of the layer.
    %       InputNames             - The names of the inputs of the layer.
    %       NumOutputs             - The number of outputs of the layer.
    %       OutputNames            - The names of the outputs of the layer.
    %
    %   Example:
    %       Create a sigmoid layer.
    %
    %       layer = SigmoidLayer()
    %
    %   Ilaria Tono, 2019, EventLab, ES
    
    properties
        % (optional) properties
    end
    
    methods
        function layer = SigmoidLayer(name)
            % Set layer name.
            if nargin > 0 
                layer.Name = name;
            else
                layer.Name = 'sigmoid_layer';
            end
            % Set layer description.
            layer.Description = 'Sigmoid function';
            layer.Type = 'Sigmoid';
        end
        function Z = predict(layer, X)
            % Forward input data through the layer and output the result.
                Z = exp(X)./(exp(X)+1);
        end
        function dLdX = backward (layer, X, Z, dLdZ, memory)
            % Back-propagate the derivative of the loss function through
            % the layer, from last to first.
            dLdX = Z .* (1-Z) .* dLdZ;
        end       
    end
end


function messageString = iGetMessageString( messageID )
messageString = getString( message( messageID ) );
end