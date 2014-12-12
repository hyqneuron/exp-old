function [tiled] = getTile(W, imgWidth, imgHeight, tilesWidth, tilesHeight)
    tiled = zeros(imgWidth * tilesWidth, imgHeight * tilesHeight);
    for y = 0:tilesHeight-1
        for x = 0:tilesWidth-1
            startX = x*imgWidth+1;
            lastX  = (x+1)*imgWidth;
            startY = y*imgHeight+1;
            lastY  = (y+1)*imgHeight;
            tiled(startY:lastY, startX:lastX) = reshape(W(y*tilesHeight+x+1,:),imgHeight, imgWidth);
        end
    end
end