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

% 1.1 auxiliary things
patches = single(data.patches8)/256;
pSize   = size(patches,1);
AvgActivation = 0.3;
AvgOff        = 1 - AvgActivation;

decay = 0.999;
antidecay=1-decay;
decay2= 0.99;
antidecay2=1-decay2;

% 1.2 logging data
endPoint = 120000;
activitiesj = zeros(endPoint,1);
activitiesk = zeros(endPoint,1);
sssj = zeros(endPoint,1);
sssk = zeros(endPoint,1);

for i = 1:endPoint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2 Activation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 3 Joint activity and weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4 Finalize
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%