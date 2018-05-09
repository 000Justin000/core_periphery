load("~/Projects/motif_CP/data/london_underground/london_underground_clean.mat");
A = zeros(size(Labelled_Network));

for i=1:size(A,1)
    for j=1:size(A,2)
        if (~isequal(A(i,j), {[0]}))
           A(i,j) = 1; 
        end
    end
end

dx = 0.05;

alphas = dx:dx:1.00;
betas  = dx:dx:1.00;

for i=1:length(alphas)
for j=1:length(betas)
[X{i,j},R(i,j)]=corepercontsharp_productnorm(A,alphas(i),betas(j));
end
end
[Y,Y1]=percentagecore3(X,R)
