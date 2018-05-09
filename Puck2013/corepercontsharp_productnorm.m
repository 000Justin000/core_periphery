function [x,f] = corepercontsharp_productnorm(A,alpha,beta)
n=length(A);

    function [p]=camel(k)
        p=-k*A*k';
    end

loss = @(m)camel(m);
s(1:n)=0;
d=floor(n*beta);
for i=1:d
    s(i)=(i/d)*(1-alpha)/2;
end
for i=(d+1):n
    s(i)=((i-d)/(n-d))*(1-alpha)/2+(1+alpha)/2;
end

sums=0;
for i=1:n
    for j=1:n
        sums=sums+s(i)*s(j);
    end
end

s=s(randperm(n))/sqrt(sums);

    function [x]=generat(x)
        a=randi(n,2,1);
        c=x(a(1));
        x(a)=x(a(2));
        x(a(2))=c;
        
    end

optionso=anneal();
optionso.Generator=@(x)(generat(x));
optionso.Verbosity=0;


[x,f]=anneal(loss,s,optionso);

end
