function [] = visP(P1,P2,size)
if nargin == 2
    W1 = P1;size=P2;
else
    W1 = log(P1./P2);
end
imagesc(reshape(W1,size,size));
colormap(jet);
axis square;
end