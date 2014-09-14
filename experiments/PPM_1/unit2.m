%{
This test uses 2 iGtor unit to test what would happen when some limited
lateral inhibition takes place.

This has to be done in several steps:
1. Initialize weights to some value
2. Compute activation value for two output units
3. Compute valus of P(ai,bj) and so on
4. Update weights
5. goto step 2
%}


%{
%%%%%%%%%%%%%%%%% 1, initialization  %%%%%%%%%%%%%%%%%
How the initialization is going to go:
%}
% patch size
patches = single(data.patches8)/256;
pSize = size(patches,1);

AvgActivation = 0.3;
AvgOff        = 1 - AvgActivation;


% Initial bias (p(on) and p(off)) of output b
Pj1 = 0.1;  % j's on
Pj2 = 1-Pj1;% off

Pk1 = 0.12; % k's on
Pk2 = 1-Pk1;% off

% four set of joint probabilities P_j_1-4
P_j_1 = AvgActivation * Pj1 * ones(pSize*pSize, 1);
P_j_2 = AvgActivation * Pj2 * ones(pSize*pSize, 1);
P_j_3 = AvgOff        * Pj1 * ones(pSize*pSize, 1);
P_j_4 = AvgOff        * Pj2 * ones(pSize*pSize, 1);
P_j = [P_j_1, P_j_2, P_j_3, P_j_4];

P_k_1 = AvgActivation * Pk1 * ones(pSize*pSize, 1);
P_k_2 = AvgActivation * Pk2 * ones(pSize*pSize, 1);
P_k_3 = AvgOff        * Pk1 * ones(pSize*pSize, 1);
P_k_4 = AvgOff        * Pk2 * ones(pSize*pSize, 1);
P_k_1(8) = 4*AvgActivation * Pk1;
P_k = [P_k_1, P_k_2, P_k_3, P_k_4];

clear P_j_1 P_j_2 P_j_3 P_j_4;
clear P_k_1 P_k_2 P_k_3 P_k_4;

% lateral connections between j and k
% we start off with no mutual inhibition whatsoever, let it evolve
P_jk_1 = Pj1*Pk1;
P_jk_2 = Pj1*Pk2;
P_jk_3 = Pj2*Pk1;
P_jk_4 = Pj2*Pk2;
P_jk = [P_jk_1, P_jk_2, P_jk_3, P_jk_4];

P_kj_1 = Pk1*Pj1;
P_kj_2 = Pk1*Pj2;
P_kj_3 = Pk2*Pj1;
P_kj_4 = Pk2*Pj2;
P_kj = [P_kj_1, P_kj_2, P_kj_3, P_kj_4];

clear P_jk_1 P_jk_2 P_jk_3 P_jk_4;
clear P_kj_1 P_kj_2 P_kj_3 P_kj_4;


decay = 0.999;
antidecay=1-decay;
decay2= 0.95;
antidecay2=1-decay2;

endPoint = 30000;
activities = zeros(endPoint,1);
sss = zeros(endPoint,1);

cP_j = zeros(pSize*pSize,4);

% we do training using patches in a sequential manner
for i = 1:endPoint
    % 2 compute activity
    input = reshape(patches(:,:,i),1,pSize*pSize);
    antinput = 1.-input;
    [activityj,sumj] = activate(input,P_j(:,1),P_j(:,2),P_j(:,3),P_j(:,4),Pj1,Pj2);
    [activityk,sumk] = activate(input,P_k(:,1),P_k(:,2),P_k(:,3),P_k(:,4),Pk1,Pk2);
    % 2.1 reactivate
    alpha = 2;%3.646150329132049;
    for j = 1:20
        activityj_tmp = activityj;
        % compute new activity for j, using activity of k
        [activityj,sumj2] = activateAgain(sumj, activityk, ...
            P_kj(1), P_kj(2), P_kj(3), P_kj(4), Pj1, Pj2, alpha);
        % compute new activity for k, using activity of j
        [activityk,sumk2] = activateAgain(sumk, activityj_tmp, ...
            P_jk(1), P_jk(2), P_jk(3), P_jk(4), Pk1, Pk2, alpha);
    end
    
    %activities(i)=activityj;
    %sss(i) = s;
    % 3 compute current P values
    input = input';
    antinput = antinput';
    cPj1 = activityj;
    cPj2 = 1 - activityj;
    cPk1 = activityk;
    cPk2 = 1 - activityk;
    
    cP_j = [input * cPj1, input * cPj2, antinput * cPj1, antinput * cPj2];
    cP_k = [input * cPk1, input * cPk2, antinput * cPk1, antinput * cPk2];

    % 4 update P values
    Pj1 = decay * Pj1 + antidecay * cPj1;
    Pj2 = decay * Pj2 + antidecay * cPj2;
    P_j = decay * P_j + antidecay * cP_j;
    
    Pk1 = decay * Pk1 + antidecay * cPk1;
    Pk2 = decay * Pk2 + antidecay * cPk2;
    P_k = decay * P_k + antidecay * cP_k;
    
    P_jk =decay2 * P_jk + antidecay2 * [cPj1*cPk1, cPj1*cPk2, cPj2*cPk1, cPj2*cPk2];
    P_kj =decay2 * P_kj + antidecay2 * [cPk1*cPj1, cPk1*cPj2, cPk2*cPj1, cPk2*cPj2];
    if mod(i,10000)==0
        fprintf('i=%d\n',i);
    end
end
P1 = P_j(:,1)/Pj1;
P2 = P_j(:,2)/Pj2;
P3 = P_j(:,3)/Pj1;
P4 = P_j(:,4)/Pj2;
figure
visP(log(P1./P2)-log(P3./P4),pSize);
P1 = P_k(:,1)/Pk1;
P2 = P_k(:,2)/Pk2;
P3 = P_k(:,3)/Pk1;
P4 = P_k(:,4)/Pk2;
figure
visP(log(P1./P2)-log(P3./P4),pSize);


