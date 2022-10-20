% a = [ones(4,2); zeros(2,2)] + rand(6,2);
% Z = linkage(a);
% figure
% dendrogram(Z)

m = in.merge;
ord = in.ord;
h = in.height;

nsmp = size(m,1);
Z = nan(nsmp,1);
im = size(nsmp,1);
for ii=1:size(m,1)
    a1 = m(ii,1);
    a2 = m(ii,2);
    
    if a1<0 && a2<0
        %tmp = [abs(a1) abs(a2) nan];
        Z(ii) = ii;
    elseif a1>0 && a2<0
        [ir,ic] = find(abs(m)>a1);
        im=im+1;
        %tmp = [abs(a1) im];
    elseif a1<0 && a2>0
        [ir,ic] = find(abs(m)>a2);
        im=im+1;
        %tmp = [abs(a2) im h(ii)];
    end
end