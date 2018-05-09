function [Y,Y1]=degreecentrality(M)

n=length(M);
for i=1:n
    Y(i)=sum(M(i,:));
end
[~,Y1]=sort(Y);
Y1=fliplr(Y1);

end