%{
This file attempts an SOM with km-style learning and softmax competition
%}

%%%%%%%%%%%%
%% Constants, input, weights
%%%%%%%%%%%%
M = 20;
N = M*M;
LRate=0.015;

numEpochs=10;
numRepeats=4;
Spreads = [200 100 50 20 10 5 3 2 1 0.5];

source = single(data.patches8)/256;
pSize   = size(source,1);
numSample= size(source,3);
numInput = pSize * pSize;
patches = reshape(source, numInput, numSample);
batchSize = 100;

W = ones(N, numInput)/sqrt(numInput)+0.1*randn(N, numInput);% We use Wx to compute input value
C = zeros(N,N); % competitive kernel used for competitive softmax
S = zeros(N,N); % similarity kernel used for collaborative learning
DistanceSqr = getDistanceSqrMatrix(M);



%%%%%%%%%%%%
%% Training
%%%%%%%%%%%%
for epoch = 1:numEpochs
    StrengthCount = zeros(N,1);
    %C = getTrapezoidKernel(DistanceSqr,7);
    S = getGaussianKernel(DistanceSqr,Spreads(epoch));
    for repeat = 1:numRepeats
    for i = 1:batchSize:numSample % minibatch
        iFirst = i;
        iLast  = i+batchSize-1;
        W = normr(W+0.000001);
        Similarity = W*W';
        C = getSigmoidKernel(Similarity, 20, 10);
        X = normc(patches(:, iFirst:iLast) +0.000001);
        Product = W*X;
        ProductExp = exp(Product*80);
        CompetitiveFactor = ProductExp./(C*ProductExp);
        CompetitiveFactorExp = exp(CompetitiveFactor*40);
        CompetitiveFactor2   = CompetitiveFactorExp ./ (C*CompetitiveFactorExp);
        Activation = Product .* CompetitiveFactor2;
        LearningFactor = S * Activation;
        W = (1-LRate) * W  +  LRate * (LearningFactor*X')./batchSize;
        StrengthCount = StrengthCount + sum(Activation, 2);
    end % minibatch
    fprintf('epoch %d/%d, repeat %d/%d\n', epoch, numEpochs, repeat, numRepeats);
    end
    figure;visP(getTile(W,8,8,M,M),8*M); colormap(gray(256));
end% epoch
W = normr(W+0.000001);
figure;visP(getTile(W,8,8,M,M),8*M); colormap(gray(256));