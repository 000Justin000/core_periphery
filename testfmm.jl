using StatsBase;
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

mutable struct Particle
    CoM::Vector{Float64}
    m::Float64
    pot::Float64
end

#----------------------------------------------------------------
function distance_matrix(coords)
    n = size(coords,2);
    D = zeros(n,n);

    for j in 1:n
        println("distance_matrix: ", j);
        for i in j+1:n
            D[i,j] = haversine(coords[:,i], coords[:,j], 6371e3)
        end
    end

    D = D + D';

    @assert issymmetric(D);

    return D;
end
#----------------------------------------------------------------


function CoM2(coord1, coord2, m1=1.0, m2=1.0)
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

function fill_cm!(cmp, idx, bt, ms, roid, srid)
    if (idx > bt.tree_data.n_internal_nodes)
        cmp[idx] = Particle(convert(Vector{Float64}, bt.hyper_spheres[idx].center), ms[roid[srid[idx]]], 0.0)
    else
        fill_cm!(cmp, idx*2,   bt, ms, roid, srid)
        fill_cm!(cmp, idx*2+1, bt, ms, roid, srid)

        cmp[idx] = Particle(CoM2(cmp[idx*2].CoM,cmp[idx*2+1].CoM, cmp[idx*2].m,cmp[idx*2+1].m), cmp[idx*2].m + cmp[idx*2+1].m, 0.0)
    end
end

function fill_p2!(cmp, idx_1, idx_2, bt)
    if ((idx_1 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs) &&
        (idx_2 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs))
        #-----------------------------------------------------------------
        distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center)
        sp1r = bt.hyper_spheres[idx_1].r
        sp2r = bt.hyper_spheres[idx_2].r
        if (distance >= 2*(sp1r + sp2r))
            cmp[idx_1].pot += cmp[idx_2].m / distance
            cmp[idx_2].pot += cmp[idx_1].m / distance
            cmp[end].pot += 1
        elseif (sp1r <= sp2r)
            fill_p2!(cmp, idx_1, idx_2*2,   bt)
            fill_p2!(cmp, idx_1, idx_2*2+1, bt)
        else
            fill_p2!(cmp, idx_1*2,   idx_2, bt)
            fill_p2!(cmp, idx_1*2+1, idx_2, bt)
        end
        #-----------------------------------------------------------------
    end
end

function fill_p!(cmp, idx, bt)
    if (idx <= bt.tree_data.n_internal_nodes)
        fill_p2!(cmp, idx*2, idx*2+1, bt)
        fill_p!(cmp,  idx*2,          bt)
        fill_p!(cmp,  idx*2+1,        bt)
    end
end

function accumulate_p!(cmp, idx, bt)
    if (2 <= idx <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
        cmp[idx].pot += cmp[div(idx,2)].pot
    end

    if (idx <= bt.tree_data.n_internal_nodes)
        accumulate_p!(cmp, idx*2,   bt)
        accumulate_p!(cmp, idx*2+1, bt)
    end
end

#--------------------------------
# load airport data and location
#--------------------------------
airports_dat = readcsv("data/open_airlines/airports.dat");
num_airports = size(airports_dat,1);
no2id = Dict{Int64, Int64}();
id2no = Dict{Int64, Int64}();
id2lc = Dict{Int64, Array{Float64,1}}();
for i in 1:num_airports
    no2id[i] = airports_dat[i,1];
    id2no[airports_dat[i,1]] = i;
    id2lc[airports_dat[i,1]] = airports_dat[i,7:8];
end
#--------------------------------

#--------------------------------
W = spzeros(num_airports,num_airports);
#--------------------------------
# the adjacency matrix
#--------------------------------
routes_dat = readcsv("data/open_airlines/routes.dat");
num_routes = size(routes_dat,1);
for i in 1:num_routes
    id1 = routes_dat[i,4];
    id2 = routes_dat[i,6];
    if (typeof(id1) == Int64 && typeof(id2) == Int64 && haskey(id2lc,id1) && haskey(id2lc,id2))
        W[id2no[id1], id2no[id2]] += 1;
    end
end
#--------------------------------
W = W + W';
#--------------------------------
A = spones(sparse(W));
#--------------------------------

#--------------------------------
# compute distance and rank
#--------------------------------
coords = zeros(2,num_airports);
for i in 1:num_airports
    coords[:,i] = flipdim(id2lc[no2id[i]],1)
end
#--------------------------------
bt = BallTree(coords, Haversine(6371e3), leafsize=1);
#--------------------------------
D = distance_matrix(coords);
#--------------------------------

dist = Dict{Int64,Array{Float64,1}}(i => vec(D[:,i]) for i in 1:num_airports);

# C = vec(sum(A,1))/maximum(vec(sum(A,1)))*10;
C = vec(readcsv("results/core_score_1.csv"))
epd_real = vec(sum(StochasticCP.probability_matrix(C, D), 1));
epd      = StochasticCP_FMM.expected_degree(C, coords, CoM2, dist, 1, bt, 0.0);

function check(ratio)
    epd_real = vec(sum(StochasticCP.probability_matrix(C, D), 1));
    epd      = StochasticCP_FMM.expected_degree(C, coords, CoM2, dist, 1, bt, ratio);

    order = sortperm(epd, rev=false);
    plot(epd_real[order])
    plot!(epd[order])
    plot!(epd[order] - epd_real[order])
end

# 
# td = bt.tree_data
# ni = td.n_internal_nodes
# nl = td.n_leafs
# roid = bt.indices             # (o)riginal id of the (r)eordered data points
# orid = sortperm(roid)         # (r)eordered id of the (o)riginal data points
# 
# # (r)eordered data point id of the hyper(s)pheres
# srid = Dict((idx >= td.cross_node) ? (idx => td.offset_cross + idx) : (idx => td.offset + idx) for idx in (ni+1:ni+nl))
# # hyper(s)phere id of the (r)eordered data points
# rsid = Dict((idx >= td.cross_node) ? (td.offset_cross + idx => idx) : (td.offset + idx => idx) for idx in (ni+1:ni+nl))
# 
# cmp = Vector{Particle}(ni+nl+1)
# cmp[end] = Particle([0.0,0.0],0.0,0.0)
# 
# ms = ones(nl)
# fill_cm!(cmp, 1, bt, ms, roid, srid)
# fill_p!(cmp, 1, bt)
# accumulate_p!(cmp, 1, bt)
# 
# pots = [cmp[rsid[orid[idx]]].pot for idx in 1:nl]
