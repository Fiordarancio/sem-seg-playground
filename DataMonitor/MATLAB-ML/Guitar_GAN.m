% Free adaptation of the MATLAB GAN tutorial in order to apply it to our
% guitar dataset
%--------------------------------------------------------------------------
% To open the example, type in command line:
%   openExample('nnet/TrainGenerativeAdversarialNetworkGANExample')

clear; close all; clc;
pathSetup;

imgDir
%---------------------------------------------------------- IMPORT DATASET
% Create an image datastore containing all guitars: they are virtually
% divided in folder so that imageDataStore can easily classify them with
% the label "Guitars"
imds = imageDatastore(imgDir, ...
    'IncludeSubFolders', true, ...
    'LabelSource', 'foldernames');
% Augment the data to include random horizontal flipping, scaling, and
% resizing. We will have all images being 64x64 as in the tutorial, and
% please note that we need quadratic sizes in order to deal with a squared
% filter map. As it can be seen later, our filter size is a x a where a is 
% an integer. If this last has not square shape, a padding will be needed
% width = 25; height = 64; % NOTE: should be equal
width = 256; height = 256;
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandScale', [1,2]);
augimds = augmentedImageDatastore([width, height], imds, ...
    'DataAugmentation', augmenter);

%------------------------------------------------------- PREPARE GENERATOR
% We define a very simple convolutional network that must generate images
% given a (1,1,100) array of random values. So, it has to upscale these
% arrays in order to get a (width,height,3) array, which is a common RGB 
% image, by using  a series of transposed convolution layers. Why a
% trnasposed? Because we are upsampling, that is like skipping by one
% On these layers, batch normalization and ReLU are applied. They have 4x4
% filters with decreasing number of filters for each layers; stride of 2
% (the stride is what divides)and final 3 filter to get RGB image.
% The final layer used for classification will be a tanh one.
%
% DEBUG NOTE: when changing the width/height, or number of filters etc.
% don't forget that evary dimension ends up in a correct size. It seems
% trivial, but it is not. The formula for calculating convolution size on
% the output is:
%   newWidth = ( ( width + padWidth - kernelWidth ) / stride ) + 1
% The newHeight is evaluated similarly. Remember that roundings are needed.
% kernelWidth and kernelHeight form the filterSize, while padWidth and
% padHeight are related to the padding

filterSize = [4,4]; % squared filter 
% filterSize = [8, 8]; 
numFilters = 64; % 8 x 8 conv squares
% numFilters = 256; 
randArrayLen = 100; % len of input random array
numLatentInputs = randArrayLen;

dispPrint('Creating net Generator...');
layersGenerator = [
    imageInputLayer([1 1 numLatentInputs], ...
        'Normalization', 'none', 'Name', 'in')
    transposedConv2dLayer(filterSize, 32 * numFilters, 'Name', 'tconv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1');
    ...
    transposedConv2dLayer(filterSize, 16 * numFilters, ...
        'Name', 'tconv2', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    ...
    transposedConv2dLayer(filterSize, 8 * numFilters, ...
        'Name', 'tconv3', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    ...
    transposedConv2dLayer(filterSize, 4 * numFilters, ...
        'Name', 'tconv4', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    ...
    transposedConv2dLayer(filterSize, 2 * numFilters, ...
        'Name', 'tconv5', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    ...
    transposedConv2dLayer(filterSize, numFilters, ...
        'Name', 'tconv6', 'Stride', 2, 'Cropping', 1)
    batchNormalizationLayer('Name', 'bn6')
    reluLayer('Name', 'relu6')
    ...
    transposedConv2dLayer(filterSize, 3, ...
        'Name', 'outputGenConv', 'Stride', 2, 'Cropping', 1)
    tanhLayer('Name', 'outputGenTanh') % no other layers required before
    % DEBUG NOTE: the tangent function can cause the presence of negative
    % values, which can lead to errors during the training. For this
    % reason, we should try to use anoter one as output. 
    % Our implementation of sigmoid layer (w.r.t. tanhLayer)
%     sigmoidLayer('Name', 'outputSigmoid')
%     reluLayer('Name', 'outputGenRelu')
    ]

% We need to convert this array of layers in a graph which connects them 
% properly. The output graph is good for DAG networks.
lgraphGenerator = layerGraph(layersGenerator);
% analyzeNetwork(layersGenerator);

% Now we create the real net. In MATLAB, a dlnetwork (Deep Learning 
% network) is a class with:
% Properties:
%     Layers        - Network layers
%     Connections   - Connections between the layers
%     Learnables    - Network learnable parameters
%     State         - State of non-learnable parameters of network
% Methods:
%     forward       - Compute d.l. network output for custom training
%     predict       - Compute deep learning network output for inference
dlnetGenerator = dlnetwork(lgraphGenerator)

%--------------------------------------------------- PREPARE DISCRIMINATOR
% Now we define the network which is responsible to classify each input
% image as "true" or "fake". So, the first layer should be designed on the
% image dimensions. Hence, we follow a structure that is symmetric w.r.t.
% the previous one about filter sizes (now increasing by ^2), and in which
% we change the transposedConv2DLayers into ordinary convolution layers.
% Here, the last layer is a convolutional one that is able to make the
% netwrok outputs scalar: we need just one element to say yes/no (1/0)
scale = 0.2;    % used by the leaky relu layer. Such layer performs 
                % f(x) = x if x>= 0; f(x) = scale*x if x <0

dispPrint('Creating net Discriminator...');
layersDiscriminator = [
    imageInputLayer([width, height, 3], ...
        'Normalization', 'none', 'Name', 'in')
    convolution2dLayer(filterSize, numFilters, ...
        'Stride', 2, 'Padding', 1, 'Name', 'conv1')
    leakyReluLayer(scale, 'Name', 'lrelu1') % notice no batchNorm initially
    ...
    convolution2dLayer(filterSize, 2 * numFilters, ...
        'Stride', 2, 'Padding', 1, 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    leakyReluLayer(scale, 'Name', 'lrelu2')
    ...
    convolution2dLayer(filterSize, 4 * numFilters, ...
        'Stride', 2, 'Padding', 1, 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    leakyReluLayer(scale, 'Name', 'lrelu3')
    ...
    convolution2dLayer(filterSize, 8 * numFilters, ...
        'Stride', 2, 'Padding', 1, 'Name', 'conv4')
    batchNormalizationLayer('Name', 'bn4')
    leakyReluLayer(scale, 'Name', 'lrelu4')
    ...
    convolution2dLayer(filterSize, 16 * numFilters, ...
        'Stride', 2, 'Padding', 1, 'Name', 'conv5')
    batchNormalizationLayer('Name', 'bn5')
    leakyReluLayer(scale, 'Name', 'lrelu5')
    ...
    convolution2dLayer(filterSize, 32 * numFilters, ...
        'Stride', 2, 'Padding', 1, 'Name', 'conv6')
    batchNormalizationLayer('Name', 'bn6')
    leakyReluLayer(scale, 'Name', 'lrelu6')
    ...
    convolution2dLayer(filterSize, 1, 'Name', 'outputDiscConv') 
    % why no activation?? let's put one
%     tanhLayer('Name', 'outputDiscTanh');
]

lgraphDiscriminator = layerGraph(layersDiscriminator);
% analyzeNetwork(layersDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator)

% plot the results
% figure
% subplot(1,2,1); plot(lgraphGenerator); title('Generator');
% subplot(1,2,2); plot(lgraphDiscriminator); title('Discriminator');

%------------------------------- DEFINE MODEL GRADIENTS AND LOSS FUNCTIONS
% In the fun folder we can find an implementation of the way we get the
% gradients from a minibatch input of variable size.
% The choice of a good loss function play a strong role. Here, G and D are
% "playing" one after the other, with G trying to maximize the probability
% to "fool" D and D trying to maximize the probability of being right
% instead. This is mapped through the loss functions: as you remember from
% Optimization method, apply a minus to maximize (by default, we write the
% functions to minimize). See the tutorial page to see the full
% specification of the function

%-------------------------------------------------------- TRAINING OPTIONS
% Let's use a new form of optimization! By now, we alweays dealt with sgdm,
% but we want to change to ADAM (ADAptive Moment estimation). ADAM is an
% adaptive learning rate optimization algorithm, specifically designed for
% deep NNs. In 2014/2015, when it was first presented, seemed promising;
% however, later showed how it could produce worse solutions than sgdm.
% There is open debate about how to fill the gap between them. The
% following values are paramenters for ADAM and they have been chosen in
% order to get a good performance -> see later if some updates improve
numEpochs = 1000;
miniBatchSize = 128; % 128
augimds.MiniBatchSize = miniBatchSize;
% if the discriminator learns to discriminate between real and generated 
% images too quickly, the generator may fail to train. In order to better 
% balance the learning of D and G, set the learn rate of G to 0.0002 (2e-4)
% and the learn rate of D to 0.0001 (1e-4)
learnRateGen = 2e-4;
learnRateDisc = 1e-4;
% for each network, initialize the trailing average gradient and trailing
% average gradient-square decay rates with [] 
trailingAvgGen = [];
trailingAvgSqGen = [];
trailingAvgDisc = [];
trailingAvgSqDisc = [];
% for both networks, use a gradient decay factor of 0.5 and a squared 
% gradient decay factor of 0.999
gradientDecayFactor = 0.5;
squaredGradientDecayFactor = 0.999;
% always train with GPU if you can
executionEnv = 'gpu'; % 'auto';

%--------------------------------------------------------------- TRAINING
% In this case we use a custom training loop, though it is very basic. For
% each epoch, shuffle the datastore and loop over minibatches. For each of
% them we:
% 1) normalize the data
% 2) convert them to dlarray of elements of type single, while dimensions 
%    labels are specified as SSCB (spatial, spatial, channel, batch). This
%    is actually a tensor
% 3) generate a new random array for G
% 4) if GPU is available, convert the array properly into a gpuArray
% 5) eavaluate gradients
% 6) update the network using ADAM
% 7) display the batch every 100 iterations, just to see what's going on!
% We don't have now the beloved training-process window, but we can create
% a held-out batch of fixed 64 (1,1,100) arrays, treated as beafore.

% create a normally distributed numbers. varargin holds sizes up to tensor
ZValidation = randn(1, 1, numLatentInputs, numFilters, 'single');
dlZValidation = dlarray(ZValidation, 'SSCB');

if (strcmp(executionEnv, 'auto') && canUseGPU) || strcmp(executionEnv, 'gpu')
    dlZValidation = gpuArray(dlZValidation);
end

% train the GAN
figure
iteration = 0;
start = tic; % look! stopwatch on

% loop over epochs
dispPrint('Starting training...');
for i = 1 : numEpochs
    % reset and shuffle the datastore
    reset(augimds);
    augimds = shuffle(augimds);
    
    % loop over batches
    while hasdata(augimds)
        iteration = iteration + 1;
        data = read(augimds);
        % ignore last partial mini-batch
        if size(data, 1) < miniBatchSize
            continue
        end
        % concatenate mini-batch and generate latent inputs for G
        X = cat(4, data{:, 1}{:}); % cat(onWhichDim, A1, A2, ...)
        Z = randn(1, 1, numLatentInputs, size(X, 4), 'single');
        % normalize
        X = (single(X)/255) * 2 - 1;
        % convert minibatch into dlarry and gpuArray just in case
        dlX = dlarray(X, 'SSCB');
        dlZ = dlarray(Z, 'SSCB');
        if (strcmp(executionEnv, 'auto') && canUseGPU) || strcmp(executionEnv, 'gpu')
            dlX = gpuArray(dlX);
            dlZ = gpuArray(dlZ);
        end 
        
        % evaluate model gradients (using the deep learning version of the
        % MATLAB function eval(@funcname, funcargs...)
        [gradientGen, gradientDisc, stateGen] = dlfeval(@modelGradients, ...
            dlnetGenerator, dlnetDiscriminator, dlX, dlZ);
        dlnetGenerator.State = stateGen;
        % update parameters of D and G using ADAM
        [dlnetDiscriminator.Learnables, trailingAvgDisc, trailingAvgSqDisc] = ...
            adamupdate(dlnetDiscriminator.Learnables, gradientDisc, ...
                trailingAvgDisc, trailingAvgSqDisc, ...
                iteration, learnRateDisc, ...
                gradientDecayFactor, squaredGradientDecayFactor);
        [dlnetGenerator.Learnables, trailingAvgGen, trailingAvgSqGen] = ...
            adamupdate(dlnetGenerator.Learnables, gradientGen, ...
                trailingAvgGen, trailingAvgSqGen, ...
                iteration, learnRateGen, ...
                gradientDecayFactor, squaredGradientDecayFactor);
        
        % display batches every 100 iterations
        if mod(iteration, 100) == 0 || iteration == 1
            % generate images using the held-out generator input
            dlXGeneratedValidation = predict(dlnetGenerator, dlZValidation);
            % rescale images in the range [0 1] and display them
            I = imtile(extractdata(dlXGeneratedValidation));
            I = rescale(I);
            image(I)
            % update the tile with training progress info
            D = duration(0, 0, toc(start), 'Format', 'hh:mm:ss');
            title(['Epoch: ' num2str(i) ' - ' ...
                'Iteration ' num2str(iteration) ' - ' ...
                'Elapsed: ' char(D)]); % putting string(D) transforms into array
            drawnow % here's an example of fflush, but works mainly on plot
         end
    end
end
disp('Training finished. Check it out!'); 

%----------------------------------------------------- GENERATE NEW IMAGES
% We have successfully finished our training, so we are ready to check how 
% good our generator has become. Just make it predict something over random
% vectors that we pass as before. Again, the presece of the gpu is
% appreciated and must be handled properly

% predict 16 images -> give array of 16 rand values
Znew = randn(1, 1, numLatentInputs, 16, 'single');
dlZnew = dlarray(Znew, 'SSCB');
if (strcmp(executionEnv, 'auto') && canUseGPU) || strcmp(executionEnv, 'gpu')
    dlZNew = gpuArray(dlZnew);
end
dlXGeneratedNew = predict(dlnetGenerator, dlZnew);
% display them
figure
I = imtile(extractdata(dlXGeneratedNew));
I = rescale(I);
image(I)
title('Generated images after the training');

%------------------------------------------------------ SAVE THE WORKSPACE
t = datetime();
t.Format = 'yyyy_MM_dd__hh_mm_ss';
file = ['gan_' char(t) '.mat'];
save(fullfile(guitarResDir, file));

