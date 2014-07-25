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
N = 100;  % number of units
endPoint = 120000; % number of samples used for training
decay = 0.999;
decay2= 0.99;
antidecay=1-decay;
antidecay2=1-decay2;
AvgOn  = 0.3; % average activity of input
AvgOff = 1 - AvgOn;
Alpha=0.02;
Steps=30;

%%%%%
% 1.1 data source
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

%%%%%
% 1.2 logging data
%%%%%
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
P_BnB = Pb1 * Pb2 * ones(N,N);

% 1.4 N-unit weight differentiation
P_A_B = P_A_B .* rand(numInput, N);
P_AnB = P_AnB .* rand(numInput, N);

for i = 1:endPoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2 Activation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2.0 compute input activation 
%%%%%
    input = patches(i,:);
    term1 = log(P_A_B) - log(P_AnB);
    term2 = log( bsxfun( @minus, P_B,   P_A_B) ) ... % compute log(P(na, b))
           -log( bsxfun( @minus, 1-P_B, P_AnB) );    % compute log(P(na,nb))
    term3 = log(1-P_B) - log(P_B);
    sums  = input * (term1-term2) + sum (term2) + (numInput-1) * term3;
    acts  = 1./(1+exp(-sums)); % activities
%%%%%
% 2.1 reactivate, later
%%%%%
    Term1 = log(P_B_B) - log(P_BnB);
    Term2 = log(abs(bsxfun(@minus, P_B,   P_B_B))) ... % log(P(nbi, bj))
           -log(abs(bsxfun(@minus, 1-P_B, P_BnB)));    % log(P(nbi,nbj))
    for j = 1:N
        Term1(j,j)=0;
        Term2(j,j)=0;
    end
    for j = 1:Steps
        newSums = -Alpha*(acts * (Term1-Term2)+sum(Term2)+(N-1)*term3)+sums;
        acts  = 1./(1+exp(-newSums));
    end
%%%%%
% 2.2 data logging
%%%%%
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
    if mod(i,1000)==0
        %Alpha = Alpha-0.005;
        P_B
        fprintf('i=%d\n',i);
    end
    if mod(i,10000)==0
        fprintf('breakpoint reached.\n')
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4 Finalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for k = 1:N
    figure
    visP(term1(:,k)-term2(:,k),pSize);
end

t=Term1-Term2