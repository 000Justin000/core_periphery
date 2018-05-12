load("~/Projects/motif_CP/data/london_underground/london_underground_clean_traffic.mat");
A = zeros(size(Labelled_Network));

for i=1:size(A,1)
    for j=1:size(A,2)
        if (~isequal(Labelled_Network(i,j), {[0]}))
           A(i,j) = length(cell2mat(Labelled_Network(i,j))); 
        end
    end
end

for i=1:100
for j=1:100
[X{i,j},R(i,j)]=corepercontsharp_productnorm(A,i/100,j/100);
end
end
[Y,Y1]=percentagecore3(X,R)
