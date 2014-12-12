function [DistanceSqr] = getDistanceSqrMatrix(M)
DistanceSqr = zeros(M*M,M*M);
for ix=0:M-1
    for iy=0:M-1
        for jx=0:M-1
            for jy=0:M-1
                iIndex = ix*M + iy +1;
                jIndex = jx*M + jy +1;
                DistanceSqr(iIndex,jIndex) = (ix-jx)^2+(iy-jy)^2;
                %ix2 = ix+M;
                %iy2 = iy+M;
                %DistanceSqr(iIndex,jIndex) = min((ix-jx)^2, (ix2-jx)^2) + min((iy-jy)^2, (iy2-jy)^2);
            end
        end
    end
end
end