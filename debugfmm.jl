using MAT;
using NetworkGen;
using StochasticCP;
using StochasticCP_FMM;
using Motif;
using Colors;
using NearestNeighbors;
using Distances;
using Plots; pyplot();
using LaTeXStrings;

#----------------------------------------------------------------
function Euclidean_CoM2(coord1, coord2, m1=1.0, m2=1.0)
    return [(coord1[1]*m1+coord2[1]*m2)/(m1+m2), (coord1[2]*m1+coord2[2]*m2)/(m1+m2)];
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function Haversine_CoM2(coord1, coord2, m1=1.0, m2=1.0)
    lon1 = coord1[1]/180*pi
    lat1 = coord1[2]/180*pi
    lon2 = coord2[1]/180*pi
    lat2 = coord2[2]/180*pi

    x1, y1, z1 = cos(lat1)*cos(lon1), cos(lat1)*sin(lon1), sin(lat1)
    x2, y2, z2 = cos(lat2)*cos(lon2), cos(lat2)*sin(lon2), sin(lat2)

    x = (x1*m1+x2*m2)/(m1+m2)
    y = (y1*m1+y2*m2)/(m1+m2)
    z = (z1*m1+z2*m2)/(m1+m2)

    lon = atan2(y,x)
    hyp = sqrt(x*x+y*y)
    lat = atan2(z,hyp)

    return [lon/pi*180, lat/pi*180]
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function check(C, D, coordinates, metric, CoM2, dist_opt, ratio)
    coords = flipdim([coordinates[i][j] for i in 1:size(coordinates,1), j in 1:2]',1);
    bt = BallTree(coords, metric, leafsize=1);
    dist = Dict{Int64,Array{Float64,1}}(i => vec(D[:,i]) for i in 1:length(C));
    epd_real = vec(sum(StochasticCP.probability_matrix(C, D.^dist_opt), 1));
    epd, fmm_tree = StochasticCP_FMM.expected_degree(C, coords, CoM2, dist, dist_opt, bt, ratio);

    order = sortperm(epd_real, rev=false);
    h = plot(epd_real[order]);
    plot!(h, epd[order]);
    plot!(h, epd[order] - epd_real[order]);

    return h, fmm_tree;
end
#----------------------------------------------------------------

data = MAT.matread("results/openflight_distance3.mat")
metric = Haversine(6371e3);
CoM2 = Haversine_CoM2;
h, fmm_tree = check(data["C"], data["D"], data["coordinates"], metric, CoM2, 3, 0.15);
display(h);
