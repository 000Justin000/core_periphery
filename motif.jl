using MatrixNetworks;
using MAT;
using LightGraphs;
using GraphPlot;
using Colors;
using Compose;
using Plots; gr()

function combinatorial_Laplacian(A)
    if (!issymmetric(A))
        throw(ArgumentError("The input matrix must be symmetric."));
    else
        n = size(A,1);
    end

    D = spdiagm(vec(sum(A,1)));

    return D - A
end


function random_walk_Laplacian(A)
    if (!issymmetric(A))
        throw(ArgumentError("The input matrix must be symmetric."));
    else
        n = size(A,1);
    end

    d = vec(sum(A,1));

    P = copy(A);

    #--------------------------------------------------------------
    # if node i is isolated, then P_{ii} = 1
    #--------------------------------------------------------------
    for i in 1:n
        @assert d[i] >= 0;

        if (d[i] > 0)
            P[i,:] = P[i,:] ./ d[i];
        elseif (d[i] == 0)
            P[i,i] = 1;
        end
    end

    return P
end


function random_core_peri(n, cratio, pcc, pcp, ppp)
    # vertices = randperm(n);
    vertices = 1:n;
    core = vertices[1:Int(floor(n*cratio))];
    peri = vertices[Int(floor(n*cratio))+1:end];

    A = spzeros(n,n);

    for i in 1:n
        for j in i+1:n
            if ((i in core) && (j in core))
                A[i,j] = rand() < pcc ? 1 : 0;
            elseif ((i in peri) && (j in peri))
                A[i,j] = rand() < ppp ? 1 : 0;
            else
                A[i,j] = rand() < pcp ? 1 : 0;
            end
        end
    end

    return A + A'
end


function random_core_peri_di(n, cratio, pcc, pcp, ppp)
    # vertices = randperm(n);
    vertices = 1:n;
    core = vertices[1:Int(floor(n*cratio))];
    peri = vertices[Int(floor(n*cratio))+1:end];

    A = spzeros(n,n);

    for i in 1:n
        for j in 1:n
            if (i != j)
                if ((i in core) && (j in core))
                    A[i,j] = rand() < pcc ? 1 : 0;
                elseif ((i in peri) && (j in peri))
                    A[i,j] = rand() < ppp ? 1 : 0;
                else
                    A[i,j] = rand() < pcp ? 1 : 0;
                end
            end
        end
    end

    return A
end


data = matread("data/foodweb.mat");

# spones Create a sparse matrix with the same structure as that of "S", 
# but with every nonzero element having the value "1.0".
A = spones(data["A"]);
# A = random_core_peri(128, 0.5, 0.8, 0.8, 0.3)
G = DiGraph(A);

# bidirectional edges
B = min.(A, A');

# unidirectional edges
U = A - B;

# ----- edge -----
W0 = max.(A, A');

# ------ M_5 ------
# why is this M_5 ?
# -----------------
T1 = (U  * U ) .* U;
T2 = (U' * U ) .* U;
T3 = (U  * U') .* U;
W1 = T1 + T2 + T3;
W1 = W1 + W1';

# ---------------------------------------------------------
EL0 = eigs(random_walk_Laplacian(W0); nev=2, which=:LR)
ES0 = eigs(random_walk_Laplacian(W0); nev=1, which=:SR)
EL1 = eigs(random_walk_Laplacian(W1); nev=2, which=:LR)
ES1 = eigs(random_walk_Laplacian(W1); nev=1, which=:SR)
# ---------------------------------------------------------

SC0 = sweepcut(W0, real(ES0[2][:,end]));
SC1 = sweepcut(W1, real(ES1[2][:,end]));

S0 = bestset(SC0);
S1 = bestset(SC1);

C0 = [(i in S0 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];
C1 = [(i in S1 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];

draw(PDF("C0.pdf", 16cm, 16cm), gplot(G, nodefillc=C0));
draw(PDF("C1.pdf", 16cm, 16cm), gplot(G, nodefillc=C1));
