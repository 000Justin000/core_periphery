%%
load('../results/brightkite_distanceopt.mat');

%%
figure
worldmap([-90, 90],[180.001, 179.999]);
load coastlines;
plotm(coastlat,coastlon);
geoshow(coastlat,coastlon);
mlabel off; plabel off;

%%
coords_data = cell2mat(coords);
coordinates = reshape(coords_data, [2,51406])';

n = 1000;
cmap = jet(n);
cid = ceil((C-min(C)) / (max(C)-min(C)) * (n-1)) + 1;

vi = ((-90 < coordinates(:,1) & coordinates(:,1) < 90) & (-179.999 < coordinates(:,2) & coordinates(:,2) < 179.999));

for i = 1:n
    geoshow(coordinates(vi & cid==i,1), coordinates(vi & cid==i,2), 'linestyle','none', 'marker','o', 'MarkerSize',3, 'MarkerEdgeColor',cmap(i,:), 'MarkerFaceColor',cmap(i,:));
end