clear all;
close all;

load('celegans_delta.mat');
[X,Y] = meshgrid(d1,d2);

figure;
surf(X,Y,omg');
xlabel('delta 1');
ylabel('delta 2');
zlabel('objective');
shading interp;
view(-125,45);

figure;
surf(X,Y,doe');
xlabel('delta 1');
ylabel('delta 2');
zlabel('epsilon derivative');
shading interp;
view(-125,45);

figure;
surf(X,Y,epd_rmse');
xlabel('delta 1');
ylabel('delta 2');
zlabel('gradient rmse');
shading interp;
view(-125,45);

figure;
surf(X,Y,tog');
xlabel('delta 1');
ylabel('delta 2');
zlabel('objective time');
shading interp;
view(-125,45);

figure;
surf(X,Y,tgd');
xlabel('delta 1');
ylabel('delta 2');
zlabel('gradient time');
shading interp;
view(-125,45);
