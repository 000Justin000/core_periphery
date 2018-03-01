using MatrixNetworks;
using MAT;
using LightGraphs;
using GraphPlot;
using Colors;
using Compose;
using Plots; gr();
using Spectral;
using Motif;

data = MAT.matread("data/reachability.mat");

od = sortperm(vec(data["populations"]), rev=true);
data["A"]           = data["A"][od,od]
data["labels"]      = data["labels"][od]
data["latitude"]    = data["latitude"][od]
data["longitude"]   = data["longitude"][od]
data["populations"] = data["populations"][od]

A = spones(data["A"]);
W0 = Motif.Me1(A);
W1 = Motif.M05(A);
W2 = Motif.M13(A);

# ---------------------------------------------------------
EL0 = eigs(Spectral.random_walk_Laplacian(W0)+2*speye(W0); nev=2, which=:LR)
ES0 = eigs(Spectral.random_walk_Laplacian(W0)+2*speye(W0); nev=1, which=:SR)
EL1 = eigs(Spectral.random_walk_Laplacian(W1)+2*speye(W1); nev=2, which=:LR)
ES1 = eigs(Spectral.random_walk_Laplacian(W1)+2*speye(W1); nev=1, which=:SR)
EL2 = eigs(Spectral.random_walk_Laplacian(W2)+2*speye(W2); nev=2, which=:LR)
ES2 = eigs(Spectral.random_walk_Laplacian(W2)+2*speye(W2); nev=1, which=:SR)
# ---------------------------------------------------------

SC0 = MatrixNetworks.sweepcut(W0, real(ES0[2][:,end]));
SC1 = MatrixNetworks.sweepcut(W1, real(ES1[2][:,end]));
SC2 = MatrixNetworks.sweepcut(W2, real(EL2[2][:,end]));

S0 = MatrixNetworks.bestset(SC0);
S1 = MatrixNetworks.bestset(SC1);
S2 = MatrixNetworks.bestset(SC2);

G = LightGraphs.DiGraph(A);
C0 = [(i in S0 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];
C1 = [(i in S1 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];
C2 = [(i in S2 ? colorant"lightseagreen" : colorant"orange") for i in 1:nv(G)];
Compose.draw(PDF("C0.pdf", 16cm, 16cm), gplot(G, nodefillc=C0));
Compose.draw(PDF("C1.pdf", 16cm, 16cm), gplot(G, nodefillc=C1));
Compose.draw(PDF("C2.pdf", 16cm, 16cm), gplot(G, nodefillc=C2));
