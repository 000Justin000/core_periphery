function [Y,Y1] = percentagecore3 (X,F)
n=length(X{1,1});
F(isnan(F))=0;
Y(1:n)=0;
[alphac,betac]=size(X);

for i=1:alphac
    for j=1:betac
        LL=X{i,j};
        LL(isnan(LL))=0;
         for k=1:n
            Y(k)=Y(k)-LL(k)*F(i,j);
        end
    end
end
Y=Y/(n*alphac*betac);
b=max(Y);
Y=Y/b;
[~,Y2]=sort(Y);
for i=1:n
    Y1(i)=Y2(n+1-i);
end
end
