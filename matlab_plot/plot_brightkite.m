%%
clear; close all;

%%
load('../results/brightkite_distanceopt.mat');

%%
h = figure('Units', 'inches', 'Position', [0,0,12,9]); 
ax = worldmap([-90, 90],[180.001, 179.999]); ax.Units = 'inches';
load coastlines;
plotm(coastlat,coastlon);
geoshow(coastlat,coastlon,'Color','black');
setm(ax,'mlabelparallel',-90);
gridm off; mlabel off; plabel off;

%%
coords_data = cell2mat(coords);
coordinates = reshape(coords_data, [2,50686]);
lats = coordinates(1,:);
lons = coordinates(2,:);

n = 100;
cmap = jet(n);
cid = (ceil((C-min(C)) / (max(C)-min(C)) * (n-1)) + 1)';

for i = 1:n
    geoshow(lats(cid==i), lons(cid==i), 'DisplayType', 'point', 'marker','o', 'MarkerSize',2, 'MarkerEdgeColor',cmap(i,:), 'MarkerFaceColor',cmap(i,:));
end

%%
print('brightkite','-dpdf','-r0');

h.PaperSize = [9.3000, 6.8100];
ax.Position = [0.0000, -0.2700, 9.3000, 7.3500];
h.Position  = [0.0000,  0.0000, 9.3000, 6.8100];

% ax.Position = [0.5000, 0.1500, 9.3000, 7.3500];
% h.Position  = [0.0000,  0.0000, 9.8000, 7.5500];

print('brightkite','-dpdf','-r0');