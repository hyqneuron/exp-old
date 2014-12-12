function [kernel] = getSigmoidKernel(DotProduct, scale, shift)
kernel = 1./(1+exp(-DotProduct*scale+shift));
end