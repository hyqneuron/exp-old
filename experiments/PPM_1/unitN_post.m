%{
This version is modified from unitN such that
1. 1-sample-per-iteration is replaced with mini-batch
2. Temperature control is added
3. Reactivation is completely changed from input to output
%}

%{
This test uses N iGtor units to test what happens when a reasonable number
of units compete.

This has to be done in several steps:
1. Initialize weights to some value
2. Compute activation value for N output units
2.1. Compute lateral-interaction activity
3. Compute current joint activities and update weights, then repeat 2
4. Finalize
%}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1 Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%
% 1.1 constants
%%%%%
N = 50;  % number of units
endPoint = 100000; % number of samples used for training
epochs=8;
decay = 0.995;
antidecay=1-decay;
AvgOn  = 0.488; % average activity of input
AvgOff = 1 - AvgOn;
Alpha=1;
Steps=1;
Beta = 1*ones(1,N);
Kappa=1;
batchSize=100;

%%%%%
%% 1.1 data source
%%%%%
source = single(data.patches8)/256;
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


%% 
%{
patches = single(activitiesLv1(60001:120000,:));
numInput = size(patches,2);
%}
%%%%%
% 1.2 logging data
%%%%%
activities = zeros(endPoint,N);
sss = zeros(endPoint,N);
Ts = ones(1,N);

% 1.3 weights
% We need several set of weights
% a. P(b), 1D array
% b. P(a|b), 2D array
% c. P(a|not b), 2D array, can be replaced with P(a), though
% d. P(b_i|b_j), 2D array
% e. P(b_i|not b_j), 2D array
Pb1 = 0.1;
Pb2 = 1-Pb1;
Initializer = 0.5;
P_B   = Initializer * Pb1 * ones(1,N);
%PnB   = Initializer * Pb2 * ones(1,N);
P_A_B = Initializer * Pb1 * AvgOn * ones(numInput, N);
P_AnB = Initializer * Pb2 * AvgOn * ones(numInput, N);
%PnA_B = Initializer * Pb1 * AvgOff* ones(numInput, N);
%PnAnB = Initializer * Pb2 * AvgOff* ones(numInput, N);

% 1.4 N-unit weight differentiation
P_A_B = P_A_B .* rand(numInput, N);
P_AnB = P_AnB .* rand(numInput, N);

Ts = ones(1,N);

for epoch = 1:epochs
for i = 1:batchSize:endPoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2 Activation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.0 compute input activation 
%%%%%
    input = patches(i:i+batchSize-1,:);
    term1 = log(P_A_B) - log(P_AnB);
    term2 = log( bsxfun(@minus, P_B,    P_A_B )) ... % compute log(P(na, b))
           -log( bsxfun(@minus, (1-P_B),P_AnB ));    % compute log(P(na,nb))
    term3 = log(1-P_B) - log(P_B);
    weight = term1-term2;
    if epoch>1
        weight = weight .* (abs(weight)>0.1);
    end
    sumsInput  = bsxfun(@rdivide, bsxfun(@plus, input * weight, sum (term2)+(numInput-1) * term3), Ts);
    Tcontrib = mean(abs(sumsInput));
    Ts = decay * Ts + antidecay * Tcontrib *1;
    sumsInput = bsxfun(@rdivide, sumsInput, Ts);
    acts  = 1./(1+exp(-10*sumsInput)); % activities
%%%%%
% 2.1 reactivate
%%%%%
    % first compute their overlap
    weight = bsxfun(@rdivide, weight, sqrt(sum(weight.^2)));
    overlap = abs(weight' * weight);
    % for each sample, compute normalized output
    % normalized output = output * (act / summed act)
    acts_exp = acts;%exp(acts*100);
    summed_contrib = acts_exp * overlap;
    actsold = acts;
    acts = actsold.* (acts_exp ./ summed_contrib).^2;
%%%%%
% 2.2 data logging
%%%%%
    activities(i:i+batchSize-1,:) = acts;
    sss(i:i+batchSize-1,:)        = sumsInput;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3 Joint activity and weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    P_B  = decay * P_B    + antidecay * mean(acts);
    P_A_B= decay * P_A_B  + antidecay * (input' * acts)/batchSize;
    P_AnB= decay * P_AnB  + antidecay * (input' * (1-acts))/batchSize;
    %{
    P_B  = P_B    + mean(acts);
    PnB  = PnB    + (1-mean(acts));
    P_A_B= P_A_B  + (input' * acts)/batchSize;
    P_AnB= P_AnB  + (input' * (1-acts))/batchSize;
    PnA_B= PnA_B  + ((1-input)' * acts)/batchSize;
    PnAnB= PnAnB  + ((1-input)' * (1-acts))/batchSize;
    %}
    if mod(i+batchSize-1,10000)==0
        fprintf('i=%d\n',i);
        Ts
    end
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4 Finalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:N
    figure
    visP(term1(:,k)-term2(:,k),pSize);
end