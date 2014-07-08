% view vector img as a square image with size=size
% img would be a black-white vector on the domain [0:255]
function [] = viewSqrImg(img, size)
imagec(reshape(img,size,size));
colormap(gray(256));
axis square
end