function [Y]= artcoredc(n,C,p,k,f)

% n=# nodes (100 in our case), C=coresize (.5 in our case), p=base edge
% probability (.25), k=core factor (1.1 -- 2), f=number of repetitions (100)

Y1(1:f)=0;
C=round(C*n);

for h=1:f
    
N(1:n,1:n)=0;
for i=1:C
    for j=1:(i-1)
        if rand<p*k^2
            N(i,j)=1;
            N(j,i)=1;
        end
    end
    for j=(C+1):n
        if rand<p*k
            N(i,j)=1;
            N(j,i)=1;
        end
    end
end

for i=(C+1):n
    for j=(C+1):(i-1)
        if rand<p*k
            N(i,j)=1;
            N(j,i)=1;
        end
    end
end
perm=randperm(n);
N=N(perm,perm);


[~,Z]= degreecentrality(N);


[~,L]=sort(perm);
 b=0;
b=2*length(intersect(L(1:n/2),Z(1:n/2)));

 Y1(h)=b/n;
end
Y=mean(Y1);
end



function myOwnG()

G1 = ErdosRenyi(nr_C,p1);
G2 = RandBip(nr_C,nr_P,p2);
G3 = ErdosRenyi(nr_P,p3);  % input('done G123');

G = [G1  G2
    G2' G3 ];
G = sparse(G)+0;      for i=1:n;     G(i,i)=0; end

end