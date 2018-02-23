using MatrixNetworks;
using MAT;
using LightGraphs;
using GraphPlot;
using Colors;
using Compose;
using Plots; gr();

function combinatorial_Laplacian(A)
    if (!issymmetric(A))
        throw(ArgumentError("The input matrix must be symmetric."));
    else
        n = size(A,1);
    end

    D = spdiagm(vec(sum(A,1)));

    return D - A;
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

        # P is a row stochastic matrix
        if (d[i] > 0)
            P[i,:] = P[i,:] ./ d[i];
        elseif (d[i] == 0)
            P[i,i] = 1;
        end
    end

    return P;
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

    return A + A';
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

    return A;
end

function distance_matrix(x,y)
    n = length(x);
    D = spzeros(n, n);

    for i in 1:n
        for j in 1:n
            if (i != j)
#               D[i,j] = ((x[j]-x[i])^2.0 + (y[j]-y[i])^2.0)^0.5;
                D[i,j] = ((x[j]-x[i])^2.0 + (y[j]-y[i])^2.0)^0.5;
            end
        end
    end

    @assert issymmetric(D);

    return D;
end

function probability_matrix(C,D)
    @assert issymmetric(D);

    rho = exp.(C .+ C') ./ (exp.(C .+ C') .+ D);
    rho = rho - spdiagm(diag(rho));

    @assert issymmetric(rho);

    return rho;
end

function model_fit(A,D)
    @assert issymmetric(A);
    @assert issymmetric(D);

    n = size(A,1);
    C = zeros(n);
    
    converged = false;
    while(!converged)
        C0 = copy(C);

        G = vec(sum(A-probability_matrix(C,D), 2));
        C = C + 0.001 * G;
        println(C[1:5])

        if (norm(C-C0)/norm(C) < 1.0e-6)
            converged = true;
        end
    end

    return C
end

# data = matread("data/foodweb.mat");

data = matread("data/reachability.mat");
od = sortperm(vec(data["populations"]), rev=true);

data["A"]           = data["A"][od,od]
data["labels"]      = data["labels"][od]
data["latitude"]    = data["latitude"][od]
data["longitude"]   = data["longitude"][od]
data["populations"] = data["populations"][od]

A = spones(data["A"]);
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

# ----- 2 hop -----
W2 = B*B - spdiagm(diag(B*B));

# ---------------------------------------------------------
EL0 = eigs(random_walk_Laplacian(W0)+2*speye(W0); nev=2, which=:LR)
ES0 = eigs(random_walk_Laplacian(W0)+2*speye(W0); nev=1, which=:SR)
EL1 = eigs(random_walk_Laplacian(W1)+2*speye(W1); nev=2, which=:LR)
ES1 = eigs(random_walk_Laplacian(W1)+2*speye(W1); nev=1, which=:SR)
EL2 = eigs(random_walk_Laplacian(W2)+2*speye(W2); nev=2, which=:LR)
ES2 = eigs(random_walk_Laplacian(W2)+2*speye(W2); nev=1, which=:SR)
# ---------------------------------------------------------

SC0 = sweepcut(W0, real(ES0[2][:,end]));
SC1 = sweepcut(W1, real(ES1[2][:,end]));
SC2 = sweepcut(W2, real(EL2[2][:,end]));

S0 = bestset(SC0);
S1 = bestset(SC1);
S2 = bestset(SC2);

C0 = [(i in S0 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];
C1 = [(i in S1 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];
C2 = [(i in S2 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];

draw(PDF("C0.pdf", 16cm, 16cm), gplot(G, nodefillc=C0));
draw(PDF("C1.pdf", 16cm, 16cm), gplot(G, nodefillc=C1));
draw(PDF("C2.pdf", 16cm, 16cm), gplot(G, nodefillc=C2));
