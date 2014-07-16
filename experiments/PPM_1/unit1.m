%{
This test uses a single iGtor unit to test what would happen when
a single unit is used and when no lateral competition is present.

This has to be done in several steps:
1. Initialize weights to some value
2. Compute activation value for unit b1
3. Compute valus of P(ai,b1) and so on
4. Update weights
5. goto step 2

This script relies on the function "activate" to perform step 2
It produces two histogram variables: 
1. "activities": activity of each activation
2. "sss": input value before activation, returned by "activate"
"activities" shows a histogram that's binary
"sss" shows a histogram that contains a wide Gaussian and a narrow
Laplacian
%}


%{
%%%%%%%%%%%%%%%%% 1, initialization  %%%%%%%%%%%%%%%%% I think I'm not
going to do initialization in a highly random manner. Here's what I'm gonna
do: 1.1 set all P(not a_i|b) = P(not a_i|not b), so that a off input has no
effect on b 1.2 set all P(a_i|b) = P(a_i|not b) but one particular a_i, so
that initially only a single input has effect on b. For that particular a,
set the ratio=4

As it turns out, step 1.2 is not necessary for some sort of differentiation
to happen. The uneven distribution of input will cause the unit to diverge
nevertheless, even though this happens only when Pb1 is initialized to
small values like 0.1.

Also, when multiple outputs are used, certainly some sort of initial
difference in initialization between those units should be introduced,
otherwise the outputs will forever behave in the same way.

Note: average activation of an input cell = 0.3. So for 1.1, both P values
would be 0.3. As for 1.2, make it 0.8:0.2 for now. Make b's own bias be
0.2:0.8
%}



% patch size
patches = single(data.patches8)/256;
pSize = size(patches,1);

AvgActivation = 0.3;
AvgOff        = 1 - AvgActivation;


% Initial bias (p(on) and p(off)) of output b
Pb1 = 0.1;  % on
Pb2 = 1-Pb1;% off

% four set of joint probabilities P_j_1-4
P_j_1 = AvgActivation * Pb1 * ones(pSize*pSize, 1);
P_j_2 = AvgActivation * Pb2 * ones(pSize*pSize, 1);
P_j_3 = AvgOff        * Pb1 * ones(pSize*pSize, 1);
P_j_4 = AvgOff        * Pb2 * ones(pSize*pSize, 1);
P_j = [P_j_1, P_j_2, P_j_3, P_j_4];
clear P_j_1 P_j_2 P_j_3 P_j_4;

decay = 0.999;
antidecay=1-decay;

endPoint = 120000;
activities = zeros(endPoint,1);
sss = zeros(endPoint,1);

cP_j = zeros(pSize*pSize,4);

% we do training using patches in a sequential manner
for i = 1:endPoint
    % 2 compute activity
    input = reshape(patches(:,:,i),1,pSize*pSize);
    antinput = 1.-input;
    [activity,s] = activate(input,P_j(:,1),P_j(:,2),P_j(:,3),P_j(:,4),Pb1,Pb2);
    activities(i)=activity;
    sss(i) = s;
    % 3 compute current P values
    input = input';
    antinput = antinput';
    cPb1 = activity;
    cPb2 = 1 - activity;
    cP_j = [input * cPb1, input * cPb2, antinput * cPb1, antinput * cPb2];

    % 4 update P values
    Pb1 = decay * Pb1 + antidecay * cPb1;
    Pb2 = decay * Pb2 + antidecay * cPb2;
    P_j = decay * P_j + antidecay * cP_j;
    if mod(i,10000)==0
        fprintf('i=%d\n',i);
    end
end
P1 = P_j(:,1)/Pb1;
P2 = P_j(:,2)/Pb2;
P3 = P_j(:,3)/Pb1;
P4 = P_j(:,4)/Pb2;
visP(log(P1./P2)-log(P3./P4),pSize);