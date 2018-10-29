%%
clear; close all;

%%
load('../results/brightkite_distanceopt.mat');

%%
h = figure('Units', 'inches'); 
ax = worldmap([-90, 90],[-180, 180]); ax.Units = 'inches';
setm(ax, 'MapProjection', 'braun');
load coastlines;
plotm(coastlat,coastlon);
geoshow(coastlat,coastlon,'Color','black');
setm(ax,'mlabelparallel',-90);
gridm off; mlabel off; plabel off;

h.PaperSize = [12.4400,  7.9230];
ax.Position = [ -1.5300, -2.1500, 15.5000, 12.2250];
h.Position  = [  0.0000,  0.0000, 12.4400,  7.9230];

%%
coords_data = cell2mat(coords);
coordinates = reshape(coords_data, [2,50686]);
lats = coordinates(1,:);
lons = coordinates(2,:);

n = 1000;
cmap = jet(n);
cid = (ceil((C-min(C)) / (max(C)-min(C)) * (n-1)) + 1)';

for i = 1:n
    geoshow(lats(cid==i), lons(cid==i), 'DisplayType', 'point', 'marker','o', 'MarkerSize',2, 'MarkerEdgeColor',cmap(i,:), 'MarkerFaceColor',cmap(i,:));
end

%%
print('brightkite','-dsvg','-r0');
