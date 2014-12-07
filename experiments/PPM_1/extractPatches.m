%{
  extract patches from mnist images
starting x/y: 4
ending x/y:   25
patchSize:    7
so we leave a border = 3, and do not extract from areas outside border

%}
%starting = 4;
%ending = 25;
starting=1;
ending=32;
patchSize = 16;
last = ending - patchSize + 1;
% we sample uniformly from [starting,last]

samplesPerImage = 2;

global patches;
source = trainAllRED;
patches = zeros(patchSize,patchSize, samplesPerImage * size(source,1));
reorderIndices = randperm(size(source,1)*samplesPerImage);


for i = 1:size(source,1)
    for j = 1:samplesPerImage
        x = randi([starting,last]);
        y = randi([starting,last]);
        extracted = extractPatch(source(i,:), x,y,patchSize,patchSize,32,32);
        patches(:,:,reorderIndices((i-1)*samplesPerImage+j)) = extracted; 
    end
end
