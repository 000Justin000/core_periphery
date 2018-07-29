%%
clear; close all;

%%
load('../results/livejournal0_distanceopt.mat');

%%
h = figure('Units', 'inches', 'Position', [0,0,12,9]);
ax = usamap('conus'); ax.Units = 'inches'; 
ax.Projection = 'orthographic';
states = shaperead('usastatelo', 'UseGeoCoords', true, 'Selector', {@(name) ~any(strcmp(name,{'Alaska','Hawaii'})), 'Name'});
faceColors = makesymbolspec('Polygon', {'INDEX', [1 numel(states)], 'FaceColor', 'w'}); %NOTE - colors are random
geoshow(ax, states, 'DisplayType', 'polygon', 'SymbolSpec', faceColors);
framem off; gridm off; mlabel off; plabel off;

%%
coords_data = cell2mat(coords);
coordinates = reshape(coords_data, [2,1155627])';

n = 100;
cmap = jet(n);
cid = ceil((C-min(C)) / (max(C)-min(C)) * (n-1)) + 1;

%---------------------------------------------------------------------
vi = false(size(C));
%---------------------------------------------------------------------
for i = 1:size(states,1)
    vi = vi | inpolygon(coordinates(:,1), coordinates(:,2), states(i).Lat, states(i).Lon);
end
%---------------------------------------------------------------------

for i = 1:n
    geoshow(coordinates(vi & cid==i,1), coordinates(vi & cid==i,2), 'DisplayType', 'point', 'marker','o', 'MarkerSize',2, 'MarkerEdgeColor',cmap(i,:), 'MarkerFaceColor',cmap(i,:));
end