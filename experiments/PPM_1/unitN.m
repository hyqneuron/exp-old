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

% 1.1 constants
N = 1;  % number of units
endPoint = 120000; % number of samples used for training
decay = 0.999;
decay2= 0.99;
antidecay=1-decay;
antidecay2=1-decay2;
AvgOn  = 0.3; % average activity of input
AvgOff = 1 - AvgOn;

% 1.1 data source
source = single(data.patches8)/256;
pSize   = size(source,1);
numInput = pSize * pSize;
patches = zeros(endPoint, pSize * pSize);
for i = 1:endPoint % cast 2D images to 1D vectors
    patches(i,:) = reshape(source(:,:,i),1,numInput);
end

% 1.2 logging data
activities = zeros(endPoint,N);
sss = zeros(endPoint,N);

% 1.3 weights
% We need several set of weights
% a. P(b), 1D array
% b. P(a|b), 2D array
% c. P(a|not b), 2D array, can be replaced with P(a), though
% d. P(b_i|b_j), 2D array
% e. P(b_i|not b_j), 2D array
Pb1 = 0.1;
Pb2 = 1-Pb1;
P_B   = Pb1 * ones(1,N);
P_A_B = Pb1 * AvgOn * ones(numInput, N);
P_AnB = Pb2 * AvgOn * ones(numInput, N);
P_B_B = Pb1 * Pb1 * ones(N,N);
P_BnB = Pb2 * Pb2 * ones(N,N);

% we'll handle the random initialization a bit later

for i = 1:endPoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2 Activation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%2.0 compute input activation 
    input = patches(i,:);
    term1 = log(P_A_B) - log(P_AnB);
    term2 = log( bsxfun( @minus, P_B,   P_A_B) ) ... % compute log(P(na, b))
           -log( bsxfun( @minus, 1-P_B, P_AnB) );    % compute log(P(na,nb))
    term3 = log(1-P_B) - log(P_B);
    sums  = input * (term1-term2) + sum (term2) + (numInput-1) * sum(term3);
    acts  = 1./(1+exp(-sums)); % activities
%2.1 reactivate, later
    for j = 1:1
        
    end
%2.2 data logging
    activities(i,:) = acts;
    sss(i,:)        = sums;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3 Joint activity and weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    P_B  = decay * P_B    + antidecay * acts;
    P_A_B= decay * P_A_B  + antidecay * (input' * acts);
    P_AnB= decay * P_AnB  + antidecay * (input' * (1-acts));
    P_B_B= decay2* P_B_B  + antidecay2* (acts'  * acts);
    P_BnB= decay2* P_BnB  + antidecay2* (acts'  * (1-acts));
    
    if mod(i,10000)==0
        fprintf('i=%d\n',i);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4 Finalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
visP(term1(:,1)-term2(:,1),8);