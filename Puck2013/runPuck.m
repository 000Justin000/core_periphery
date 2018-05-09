function [Y] = runPuck(A)

load rezPuckFin
return

save firstExamplePuck.mat A
Q=100;
for a=1:Q
    for b=1:Q
          disp([a b])  
        [X{a,b},R(a,b)]=corepercontsharp_productnorm(A,a/Q,b/Q);
    end
end

save rezPuck.mat X R A

[Y,Ysorted]=percentagecore3(X,R)

save rezPuckFin.mat X R A Y Ysorted

end