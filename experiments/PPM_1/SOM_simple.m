%{
This file attempts a simple SOM without probabilistic learning
%}

%%%%%%%%%%%%
%% Constants, input, weights
%%%%%%%%%%%%
M = 20;
N = M*M;
LRate=0.005;

numEpochs=10;
numRepeats=5;
Spreads = [40 30 20 10 5 3 2 1 0.5 0.2];

source = single(data.patches8)/256;
pSize   = size(source,1);
numSample= size(source,3);
numInput = pSize * pSize;
patches = reshape(source, numInput, numSample);
batchSize = 100;

W = ones(N, numInput)/sqrt(numInput)+0.1*randn(N, numInput);% We use Wx to compute input value
K = zeros(N,N); % similarity kernel used for collaborative learning
DistanceSqr = getDistanceSqrMatrix(M);



%%%%%%%%%%%%
%% Training
%%%%%%%%%%%%
for epoch = 1:numEpochs
    StrengthCount = zeros(N,1);
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
    end
    fprintf('epoch=%d\n', epoch);
end% epoch
W = normr(W);