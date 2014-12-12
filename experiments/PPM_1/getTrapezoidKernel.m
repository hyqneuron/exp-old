function [kernel] = getTrapezoidKernel(distanceSqr, spread)
distance = sqrt(distanceSqr);
condition1 = distance<=spread;
condition2 = distance>2*spread;
condition3 = not(condition1|condition2);
kernel = condition1 + condition3.*(2*spread-distance)./spread;

%kernel = condition1;
end