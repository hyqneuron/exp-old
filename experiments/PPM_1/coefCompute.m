function [termP, termFixed]=coefCompute(AonB,B1,B2)
P1 = AonB(1)/B1;
P2 = AonB(2)/B2;
P3 = AonB(3)/B1;
P4 = AonB(4)/B2;
Term1 = log(P1/P2)
Term2 = log(P3/P4)
termP = Term1-Term2;
termFixed = Term2;
end