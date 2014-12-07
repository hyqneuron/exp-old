function [activity, s] = activate(X,P_j_1,P_j_2,P_j_3,P_j_4,Pb1,Pb2, T)
% compute division, then log
P1 = P_j_1 / Pb1;
P2 = P_j_2 / Pb2;
P3 = P_j_3 / Pb1;
P4 = P_j_4 / Pb2;

W1 = log(P1./P2);
W2 = log(P3./P4);
O  = log(Pb1/Pb2);

% x*W1 + (1-x)*W2 + O = x * (W1 - W2) + W2 + O
s = X*(W1-W2)+sum(W2)+O;
%s = s./T;
%SW1 = sum(W1)
%SW2 = sum(W2)
% activity = 1/(1+e^(-s))
activity = 1./(1+exp(-s));
end