function [activity, s] = activateAgain(origSum, j, ...
                P_jk_1, P_jk_2, P_jk_3, P_jk_4, Pk1, Pk2, ...
                Alpha)
% activateAgain  second activation pass for lateral inhibition
%   An output unit is first activated using "activate", which only accepts
%   the input. After the first pass, an activity value is obtained for each
%   output. We then feed these activities and the lateral interactions to
%   "activateAgain" to simulate the lateral interaction. Theoretically a
%   set of differential equations should be solved. But we do it in a
%   somewhat heuristic and numerical manner by doing the lateral passes
%   just a few times.
%   
%   Alpha is the step size we use to control accuracy.
%   
%   We compute the effect of j on k. So we output the activity of k
P1 = P_jk_1 / Pk1;
P2 = P_jk_2 / Pk2;
P3 = P_jk_3 / Pk1;
P4 = P_jk_4 / Pk2;

% right now we work for the 2-output case, so j is not a vector but a
% single probability value. W1 and W2, P1234 are all scalars.
W1 = log(P1/P2)
W2 = log(P3/P4)
% x*W1 + (1-x)*W2 + O = x * (W1 - W2) + W2 + O
s = -Alpha* (j*(W1-W2)+W2) + origSum;
origSum = origSum
inhit = -Alpha* (j*(W1-W2)+W2)
%SW1 = sum(W1)
%SW2 = sum(W2)
% activity = 1/(1+e^(-s))
activity = 1/(1+exp(-s));
end