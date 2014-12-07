%{
This version is modified from unitN_post so that
1. the probabilistic learning approach is replaced with k-means-like method
%}

%{
This test uses N iGtor units to test what happens when a reasonable number
of units compete.

This has to be done in several steps:
1. Initialize constants, data source, and (logging data, joint activity)
2.1 Compute activation value for N output units
2.2 Compute lateral-interaction activity
2.3 (log data, update joing activity)
3. Repeat 2
4. Finalize
%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1 Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%
% 1.1 constants
%%%%%
N = 40;  % number of units
endPoint = 100000; % number of samples used for training
epochs=4;
decay = 0.995;
antidecay=1-decay;
batchSize=100;

%%%%%
%% 1.2 data source
%%%%%
source = single(cifar10data.patches8)/256;
pSize   = size(source,1);
numInput = pSize * pSize;
patches = zeros(endPoint, pSize * pSize);
for i = 1:endPoint % cast 2D images to 1D vectors
    if mod(i,2)==0
        patches(i,:) = reshape(source(:,pSize:-1:1,i),1,numInput);
    else
        patches(i,:) = reshape(source(:,:,i),1,numInput);
    end
end
% hist equalization and zero-centering
patches = histeq(patches, 100);
patches = bsxfun(@minus, patches, mean(patches));

%%
%%%%%
% 1.3 logging data and joint activities
%%%%%
preacts    = zeros(endPoint,N);
activities = zeros(endPoint,N);
sss = zeros(endPoint,N);
XY = 0.1*rand(numInput, N);

filter = [0, 0.05, 0.08, 0.1, 0.12, 0.13, 0.15, 0.18];

for epoch = 1:epochs
for i = 1:batchSize:endPoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2 Activation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.1 compute input activation 
%%%%%
    input = patches(i:i+batchSize-1,:);
    % normalize weight
    weight = normc(XY)*1;
    weight = weight .* (abs(weight)>filter(epoch));
    weight = normc(weight);
    sumsInput  = input * weight./1.3;
    acts  = 1./(1+exp(-sumsInput)); % activities
%%%%%
% 2.2 reactivate
%%%%%
    % first compute their overlap
    overlap = abs(weight' * weight);
    % for each sample, compute normalized output
    % normalized output = output * (act / summed act)
    acts_exp = exp(acts*100);
    summed_contrib = acts_exp * overlap;
    actsold = acts;
    acts = actsold.* (acts_exp ./ summed_contrib);
%%%%%
% 2.3 (data logging, joint activity)
%%%%%
    preacts(i:i+batchSize-1,:) = actsold;
    activities(i:i+batchSize-1,:) = acts;
    sss(i:i+batchSize-1,:)        = sumsInput;
    
    XY = decay * XY + antidecay * (input' * acts);
    
    if mod(i+batchSize-1,10000)==0
        fprintf('i=%d\n',i+batchSize-1);
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4 Finalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:N
    figure
    visP(normc(XY(:,k)),pSize);
end