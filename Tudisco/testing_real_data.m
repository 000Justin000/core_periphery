clear all
close all
clc
addpath('functions');

verbose = true; 

cities = {'cities/Edinburgh', 'cities/Glasgow', 'cities/Cardiff', 'cities/Bristol', 'cities/Nottingham'};

wsdm_datasets = ["wsdm_celegans", "wsdm_london", "wsdm_pvmiun42d1", "wsdm_openflights", "wsdm_brightkite"];

NAME = 'network';
dataset_name = 'london'; 
               %'netscience'; 
               %'Erdos971';
               %'yeast'; 
               % cities{3}; 
datafilename = sprintf('./Datasets/wsdm/%s',wsdm_datasets(2));
load(datafilename);


% A = Problem.A;
A = spones(A);
A2 = A;

G = graph(A);
ind = max_connected_component(G);
A = A(ind, ind);


if verbose
   fprintf('#### %s ####', NAME);
   fprintf('\nNodes:\t%d',length(A));
   fprintf('\nEdges:\t%d',nnz(A)/2);
   fprintf('\n\n');
end

%%% NONLINEAR CP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 60; pnorm = 2*alpha; 
[x, x_array, er_array] = maximize_falpha(A,alpha,pnorm, 'verbose', verbose);
if verbose, fprintf('\n'); end
x = linearize(x,0,1);

ll = zeros(31,101);
for i = 0:30
    for j = 0:100
        ll(i+1,j+1) = log_likelihood(A,x,i*0.1,j*0.1);
    end
end

[~,ind] = sort(x,'descend');
figure, spy(A(ind,ind)); title('NSM'); setplotstuff();


%% DEGREE CP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
d = sum(A);
d = linearize(d,0,1);   

[~,ind] = sort(d,'descend');
figure, spy(A(ind,ind)); title('Degree'); setplotstuff();


function ll = log_likelihood(A,x,t,s)
    [~,rk]=ismember(x,sort(x,'descend'));
    irk = size(A,1)-rk;
    P = sigmoid(max(irk*ones(1,length(irk)), ones(length(irk),1)*irk')/length(irk), t, s);
    
    ll = sum(sum(triu(log(P),1) .* triu(A,1))) + sum(sum(triu(log(1-P),1) .* triu(1-A,1)));
end

function [ind] = max_connected_component(G)
    bins = conncomp(G);
    labels = unique(bins);
    maxval = 0; maxindex = 1;
    for i = 1 : length(labels)
        val = sum(bins==labels(i));
        if val > maxval, maxval = val; maxindex = i; end
    end
    ind = (bins==maxindex);
end
     

% interpolates (linearly) the values of x so to span the range [a,b] 
function xx = linearize(x,a,b)
    alpha = (b-a)./(max(max(x))-min(min(x))); beta = alpha*min(min(x))-a; xx = alpha.*x-beta;
end


function [] = setplotstuff(x,y) 
if nargin<2, y=1; end
if nargin<1, x=1; end

set(gca, ...
  'Box'         , 'on'          , ...
  'PlotBoxAspectRatio',[x y 1]  , ...
  'TickDir'     , 'in'          , ...
  'TickLength'  , [.01 .01]     , ...
  'XMinorTick'  , 'off'         , ...
  'YMinorTick'  , 'off'         , ...
  'YGrid'       , 'off'         , ...
  'XGrid'       , 'off'         , ...
  'XColor'      , [.3 .3 .3]    , ...
  'YColor'      , [.3 .3 .3]    , ...
  'LineWidth'   , 2           , ...        
  'FontSize'    , 13);

set(gca,'xticklabel','');
set(gca,'yticklabel','');
xlabel('');
ylabel('');
end