%{
  extract patches from mnist images
starting x/y: 4
ending x/y:   25
patchSize:    7
so we leave a border = 3, and do not extract from areas outside border

%}
starting = 4;
ending = 25;
patchSize = 15;
last = ending - patchSize + 1;
% we sample uniformly from [starting,last]

samplesPerImage = 2;

global patches;
patches = zeros(patchSize,patchSize, samplesPerImage * size(trainAll,1));

for i = 1:size(trainAll,1)
    for j = 1:samplesPerImage
        x = randi([starting,last]);
        y = randi([starting,last]);
        extracted = extractPatch(trainAll(i,:), x,y,patchSize,patchSize,28,28);
        patches(:,:,(i-1)*samplesPerImage+j) = extracted; 
    end
end
