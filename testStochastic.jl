using MAT;
using NetworkGen;
using StochasticCP;
using Motif;
using Colors;
using NearestNeighbors
using Plots; gr();

#----------------------------------------------------------------
function test_n_r_p_k(n, cratio, p, klist=1.00:0.05:2.00, repeat=1)
    crs = [];
    for k in klist
        #--------------------------------------------------------
        # A  = NetworkGen.n_r_cc_cp_pp(n, cratio, k^2*p, k*p, k*p);
        #--------------------------------------------------------
        cr = 0;
        #--------------------------------------------------------
        for itr in 1:repeat
            A  = NetworkGen.n_r_cc_cp_pp(n, cratio, k^2*p, k*p, k*p);
            C   = StochasticCP.model_fit(A);
            od  = sortperm(C, rev=true);
            cr += (1/repeat) * sum([i<=n*cratio ? 1 : 0 for i in od[1:Int(n*cratio)]])/(n*cratio);
        end
        #--------------------------------------------------------
        push!(crs, cr);
        #--------------------------------------------------------
    end
    plot(klist, crs);
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_lattice(n)
    A = NetworkGen.periodic_lattice(n,n);
    C = StochasticCP.model_fit(A);

    plot(C);
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_reachability()
    data = MAT.matread("data/benson/reachability.mat");
    
    od = sortperm(vec(data["populations"]), rev=true);
    data["A"]           = data["A"][od,od]
    data["labels"]      = data["labels"][od]
    data["latitude"]    = data["latitude"][od]
    data["longitude"]   = data["longitude"][od]
    data["populations"] = data["populations"][od]

    A  = spones(data["A"]);
    W0 = Motif.Me1(A);
    C  = StochasticCP.model_fit(W0);
    W1 = StochasticCP.model_gen(C);

    plot(C);

    return W0, W1, data;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function dist_earth(coord1, coord2)
    lat1 = coord1[1]/180 * pi;
    lon1 = coord1[2]/180 * pi;
    lat2 = coord2[1]/180 * pi;
    lon2 = coord2[2]/180 * pi;

    dlat = lat2-lat1;
    dlon = lon2-lon1;

    haversine = sin(dlat/2)^2 + cos(lat1)*cos(lat2)*sin(dlon/2)^2;
    d12 = 6371e3 * (2 * atan2(sqrt(haversine), sqrt(1-haversine)));

    return d12
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function distance_matrix(coords)
    n = size(coords,1);
    D = spzeros(n,n);

    for i in 1:n
        println(i);
        for j in i+1:n
            D[i,j] = dist_earth(coords[i], coords[j]);
        end
    end

    D = D + D';

    @assert issymmetric(D);

    return D;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function rank_distance_matrix(D)
    @assert issymmetric(D);

    n = size(D,1);

    R = zeros(D);
    for i in 1:n
        od = sortperm(D[i,:]);
        for j in 2:n
            R[i,od[j]] = j-1;
        end
    end

    R = min.(R,R');

    return R;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_underground(distance_option="no_distance")
    data = MAT.matread("data/london_underground/london_underground_clean.mat");

    W = [Int(sum(list .!= 0)) for list in data["Labelled_Network"]];
    A = spones(sparse(W));

    D = distance_matrix(data["Tube_Locations"]);

    if (distance_option == "no_distance")
        C = StochasticCP.model_fit(A);
        B = StochasticCP.model_gen(C);
    elseif (distance_option == "distance")
        C = StochasticCP.model_fit(A, D);
        B = StochasticCP.model_gen(C, D);
    elseif (distance_option == "rank_distance")
        C = StochasticCP.model_fit(A, rank_distance_matrix(D));
        B = StochasticCP.model_gen(C, rank_distance_matrix(D));
    else
        error("distance_option not supported");
    end

    return A, B, C, data;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_underground(A, C, data, option="degree", filename="output")
    @assert issymmetric(A);
    n = size(A,1);

    D = vec(sum(A,1));
    
    if (option == "degree")
        color = [(i in sortperm(D, rev=true)[1:60] ? colorant"orange" : colorant"blue") for i in 1:n];
    elseif (option == "core_score")
        color = [(i in sortperm(C, rev=true)[1:60] ? colorant"orange" : colorant"blue") for i in 1:n];
    else
        error("option not supported.");
    end

    if (option == "degree")
        ms = D;
    elseif (option == "core_score")
        ms=(2.^C/maximum(2.^C))*5;
    else
        error("option not supported.");
    end

    coords = data["Tube_Locations"];
    plot();
    for i in 1:n
        for j in i+1:n
            if (A[i,j] != 0)
                plot!([coords[i][1], coords[j][1]], [coords[i][2], coords[j][2]], leg=false, color="black");
            end
        end
    end
    scatter!([coord[1] for coord in coords], [coord[2] for coord in coords], ms=ms, c=color);
    png("results/" * filename);
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_openflight()
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
    # compute distance and rank
    #--------------------------------
    coordinates = [];
    for i in 1:num_airports
        push!(coordinates, id2lc[no2id[i]])
    end
    coordinates = [coordinates[i][j] for i in 1:size(coordinates,1), j in 1:2]';
    #--------------------------------
    kdt = NearestNeighbors.KDTree(coordinates)
    #--------------------------------

    #--------------------------------
    # the weighted adjacency matrix
    #--------------------------------
    dist_array = [];
    rank_array = [];
    #--------------------------------
    W = spzeros(num_airports, num_airports);
    routes_dat = readcsv("data/open_airlines/routes.dat");
    num_routes = size(routes_dat,1);
    for i in 1:num_routes
        id1 = routes_dat[i,4];
        id2 = routes_dat[i,6];
        if (typeof(id1) == Int64 && typeof(id2) == Int64 && haskey(id2lc,id1) && haskey(id2lc,id2))
            W[id2no[id1], id2no[id2]] += 1;
            push!(dist_array, dist_earth(id2lc[id1], id2lc[id2]));
            push!(rank_array, size(inrange(kdt, id2lc[id1], norm(id2lc[id1] - id2lc[id2])), 1));
            push!(rank_array, size(inrange(kdt, id2lc[id2], norm(id2lc[id1] - id2lc[id2])), 1));
        end
    end
    #--------------------------------

    return dist_array, rank_array
end
#----------------------------------------------------------------
