%{
This file attempts a probabilistic SOM
%}

%%%%%%%%%%%%
%% Constants, input, weights
%%%%%%%%%%%%
M = 10;
N = M*M;
LRate = 0.01;

numEpochs=1;
numRepeats=1;
Spreads = [40];

source = single(data.patches8)/256;
pSize   = size(source,1);
numSample= size(source,3);
numInput = pSize * pSize;
patches = reshape(source, numInput, numSample);
batchSize = 100;


W = ones(N, numInput)/sqrt(numInput)+0.1*randn(N, numInput);% We use Wx to compute input value
C = zeros(N,N); % competitive kernel used for competitive softmax
K = zeros(N,N); % similarity kernel used for collaborative learning
DistanceSqr = getDistanceSqrMatrix(M);

PbOn = 0.1; % should be somewhere around average number of activation per map / N
PbOff=1-PbOn;

%%%%%%%%%%%%
%% Training
%%%%%%%%%%%%
for epoch = 1:numEpochs
    StrengthCount = zeros(N,1);
    C = exp(-DistanceSqr./(Spreads(epoch)*1.5));
    K = exp(-DistanceSqr./Spreads(epoch));
    for repeat = 1:numRepeats
    for i = 1:batchSize:numSample % minibatch
        iFirst = i;
        iLast  = i+batchSize-1;
        W = normr(W);
        X = normc(patches(:, iFirst:iLast));
        Product = W*X;
        [MaxVals, MaxIs] = max(Product);
        W = (1-LRate) * W  +  LRate * ( X * K(MaxIs,:))' ./ batchSize;
        for j = 1:batchSize
            StrengthCount(MaxIs(j)) = StrengthCount(MaxIs(j))+1;
        end
    end % minibatch
    end % repeat
    fprintf('epoch=%d/%d, repeat %d/%d\n', epoch, numEpochs, repeat, numRepeats);
end% epoch

%%%%%%%%%%%
%% Finalize
%%%%%%%%%%%