% extract a single patch from an image
% x,y are the starting position
% dx,dy are the size of the patch
% origx,origy are the original size of the image
function [extracted] = extractPatch(img, x,y,dx,dy, origx,origy)
reshaped= reshape(img,origx,origy);
extracted = reshaped(x:dx+x-1,y:dy+y-1);
end