function [kernel] = getGaussianKernel(distanceSqr, spread)
kernel = exp(-distanceSqr./spread);
end