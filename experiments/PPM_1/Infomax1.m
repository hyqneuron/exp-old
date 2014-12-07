% this is an attempt to use Bell and Sejnowski's infomax code on MNIST
% patches. Weights are randomly initialized around 0

source = cifar10data.patches8;
P = size(source,3);
N = size(source,1)*size(source,2);
M = 128;
B = 100;
L = 0.001;

X = reshape(source, 64, 100000)/256;
%{
    Xmean = mean(X,2);
    Xshift= bsxfun(@minus, X, Xmean);
    Xcov = Xshift * Xshift';
    [V sig] = svd(Xcov);
    for i=1:N
        V(:,i) = V(:,i)./sqrt(sig(i,i));
    end
    X = V'*Xshift;
%}
W = zeros(M, N)+0.1*(rand([M N])-0.5);
m = randperm(N);
for i=1:M
    W(i,m(mod(i,N)+1))=1;
end
sweep = 0;
BI = B * eye(M);


wt = W;
deltaMax = 10;

while deltaMax>0.000001
    sweep=sweep+1; t=1;
    noblocks=fix(P/B);
    for t=t:B:t-1+noblocks*B,
        u=W*X(:,t:t+B-1);
        y=1./(1+exp(-u));
        W=W+L*(BI+(1-2*y)*u')*W;
    end
    deltaMax=max(max(abs(wt-W)))
    wt=W;
end