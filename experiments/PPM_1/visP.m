function [] = visP(P1,P2,size)
global cmap2
if nargin == 2
    W1 = P1;size=P2;
else
    W1 = log(P1./P2);
end
image(reshape(W1,size,size)*20+100);
colormap(cmap2);
axis square;
end