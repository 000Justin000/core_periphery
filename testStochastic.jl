using StatsBase;
using MAT;
using Colors;
using Plots; pyplot();
using LaTeXStrings;
using MatrixNetworks;
using Dierckx;
using Distances;
using NearestNeighbors;
using StochasticCP;
using StochasticCP_SGD;
using StochasticCP_FMM;

# #----------------------------------------------------------------
# function test_n_r_p_k(n, cratio, p, klist=1.00:0.05:2.00, repeat=1)
#     crs = [];
#     for k in klist
#         #--------------------------------------------------------
#         # A  = NetworkGen.n_r_cc_cp_pp(n, cratio, k^2*p, k*p, k*p);
#         #--------------------------------------------------------
#         cr = 0;
#         #--------------------------------------------------------
#         for itr in 1:repeat
#             A  = NetworkGen.n_r_cc_cp_pp(n, cratio, k^2*p, k*p, k*p);
#             C   = StochasticCP.model_fit(A);
#             od  = sortperm(C, rev=true);
#             cr += (1/repeat) * sum([i<=n*cratio ? 1 : 0 for i in od[1:Int(n*cratio)]])/(n*cratio);
#         end
#         #--------------------------------------------------------
#         push!(crs, cr);
#         #--------------------------------------------------------
#     end
#     plot(klist, crs);
# end
# #----------------------------------------------------------------
#
# #----------------------------------------------------------------
# function test_lattice(n)
#     A = NetworkGen.periodic_lattice(n,n);
#     C, epsilon = StochasticCP.model_fit(A);
#
#     plot(C);
# end
# #----------------------------------------------------------------
#
# #----------------------------------------------------------------
# function test_reachability()
#     data = MAT.matread("data/benson/reachability.mat");
#
#     od = sortperm(vec(data["populations"]), rev=true);
#     data["A"]           = data["A"][od,od]
#     data["labels"]      = data["labels"][od]
#     data["latitude"]    = data["latitude"][od]
#     data["longitude"]   = data["longitude"][od]
#     data["populations"] = data["populations"][od]
#
#     A  = spones(data["A"]);
#     W0 = Motif.Me1(A);
#     C  = StochasticCP.model_fit(W0);
#     W1 = StochasticCP.model_gen(C);
#
#     plot(C);
#
#     return W0, W1, data;
# end
# #----------------------------------------------------------------

#----------------------------------------------------------------
function Euclidean_CoM2(coord1, coord2, m1=1.0, m2=1.0)
    if (m1 == 0.0 && m2 == 0.0)
        m1 = 1.0;
        m2 = 1.0;
    end

    return (coord1*m1+coord2*m2)/(m1+m2);
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function Haversine_CoM2(coord1, coord2, m1=1.0, m2=1.0)
    if (m1 == 0.0 && m2 == 0.0)
        m1 = 1.0;
        m2 = 1.0;
    end

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
function Hamming_CoM2(coord1, coord2, m1=1.0, m2=1.0)
    return m1 >= m2 ? coord1 : coord2;
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

    return d12;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function Euclidean_matrix(coords)
    n = size(coords,1);
    D = zeros(n,n);

    for j in 1:n
        for i in j+1:n
            D[i,j] = euclidean(coords[i], coords[j])
        end
    end

    D = D + D';

    @assert issymmetric(D);

    return D;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function Haversine_matrix(coordinates)
    n = size(coordinates,1);
    D = zeros(n,n);

    for j in 1:n
        for i in j+1:n
            D[i,j] = haversine(flipdim(coordinates[i],1), flipdim(coordinates[j],1), 6371e3)
        end
    end

    D = D + D';

    @assert issymmetric(D);

    return D;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function Hamming_matrix(coordinates)
    n = size(coordinates,1);
    D = zeros(n,n);

    for j in 1:n
        for i in j+1:n
            D[i,j] = hamming(coordinates[i], coordinates[j]);
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
    for j in 1:n
        od = sortperm(D[:,j]);
        for i in 2:n
            R[od[i],j] = i-1;
        end
    end

    R = min.(R,R');

    return R;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_core_periphery(h, A, C, coords, option="degree";
                             plot_links=false,
                             distance="Euclidean")
    @assert issymmetric(A);
    n = size(A,1);

    d = vec(sum(A,1));

    if (option == "degree")
        color = [(i in sortperm(d)[end-Int64(ceil(0.03*n)):end] ? colorant"orange" : colorant"blue") for i in 1:n];
    elseif (option == "core_score")
        color = [(i in sortperm(C)[end-Int64(ceil(0.03*n)):end] ? colorant"orange" : colorant"blue") for i in 1:n];
    else
        error("option not supported.");
    end

    if (option == "degree")
        println("option: degree")
        rk = sortperm(sortperm(d))
        ms = (rk/n).^20 * 6 + 1.5;
    elseif (option == "core_score")
        println("option: core_score")
        rk = sortperm(sortperm(C))
        ms = (rk/n).^20 * 6 + 1.5;
    else
        error("option not supported.");
    end

    #------------------------------------------------------------
    if (plot_links == true)
        #------------------------------------------------------------
        if (distance == "Euclidean" && length(coords[1]) == 2)
            #--------------------------------------------------------
            for i in 1:n
                for j in i+1:n
                    #------------------------------------------------
                    if (A[i,j] != 0)
                        plot!(h, [coords[i][1], coords[j][1]],
                                 [coords[i][2], coords[j][2]],
                                 legend=false,
                                 color="black",
                                 linewidth=0.10,
                                 alpha=1.00);
                    end
                    #------------------------------------------------
                end
            end
            #--------------------------------------------------------
        elseif (distance == "Haversine")
            #--------------------------------------------------------
            for i in 1:n
                for j in i+1:n
                    #------------------------------------------------
                    if (A[i,j] != 0)
                        if (abs(coords[i][1] - coords[j][1]) <= 180)
                            plot!(h, [coords[i][1], coords[j][1]],
                                     [coords[i][2], coords[j][2]],
                                     legend=false,
                                     color="black",
                                     linewidth=0.10,
                                     alpha=0.15);
                        else
                            min_id = coords[i][1] <= coords[j][1] ? i : j;
                            max_id = coords[i][1] >  coords[j][1] ? i : j;

                            lat_c  = ((coords[min_id][2] - coords[max_id][2]) / ((coords[min_id][1] + 360) - coords[max_id][1])) * (180 - coords[max_id][1]) + coords[max_id][2]

                            plot!(h, [-180.0, coords[min_id][1]],
                                     [lat_c,  coords[min_id][2]],
                                     legend=false,
                                     color="black",
                                     linewidth=0.10,
                                     alpha=0.15);

                            plot!(h, [coords[max_id][1], 180.0],
                                     [coords[max_id][2], lat_c],
                                     legend=false,
                                     color="black",
                                     linewidth=0.10,
                                     alpha=0.15);
                        end
                    end
                    #------------------------------------------------
                end
            end
            #--------------------------------------------------------
        else
            error("distance not supported.");
        end
        #------------------------------------------------------------
    end
    #----------------------------------------------------------------
    scatter!(h, [coord[1] for coord in coords[rk]], [coord[2] for coord in coords[rk]], ms=ms[rk], c=color[rk], alpha=1.00, label="");
    #----------------------------------------------------------------
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_underground(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=10000)
    data = MAT.matread("data/london_underground/london_underground_clean.mat");

    W = [Int(sum(list .!= 0)) for list in data["Labelled_Network"]];
    A = spones(sparse(W));

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;

    #---------------------------------------------------------------------------------------------
    # if epsilon is integer, then fix epsilon, otherwise optimize epsilon as well as core_score
    #---------------------------------------------------------------------------------------------
    if (epsilon > 0)
        D = Haversine_matrix(data["Tube_Locations"]);
        C, epsilon = StochasticCP.model_fit(A, D, epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    elseif (epsilon < 0)
        D = rank_distance_matrix(Haversine_matrix(data["Tube_Locations"]));
        C, epsilon = StochasticCP.model_fit(A, D, -epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    else
        D = ones(A)-eye(A);
        C, epsilon = StochasticCP.model_fit(A, D, 1; opt=opt);
        B = StochasticCP.model_gen(C, D, 1);
    end

    return A, B, C, D, data["Tube_Locations"], epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_underground(A, C, coords, option="degree", filename="output")
    h = plot(size=(450,350), title="London Underground",
                             xlabel=L"\rm{Longitude}(^\circ)",
                             ylabel=L"\rm{Latitude}(^\circ)",
                             framestyle=:box);

    plot_core_periphery(h, A, C, [flipdim(coord,1) for coord in coords], option;
                        plot_links=true,
                        distance="Haversine")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function hist_openflight_probE(A, B, D)
    @assert issymmetric(A);
    @assert issymmetric(B);
    @assert issymmetric(D);

    n = size(A,1);
    AS = Array{Float64,1}();
    BS = Array{Float64,1}();
    DS = Array{Float64,1}();

    #------------------------------------------------------------
    for i in 1:n
        for j in i+1:n
            #----------------------------------------------------
            if (A[i,j] == 1)
                push!(AS, D[i,j]);
            end
            #----------------------------------------------------
            if (B[i,j] == 1)
                push!(BS, D[i,j]);
            end
            #----------------------------------------------------
            push!(DS, D[i,j]);
            #----------------------------------------------------
        end
    end
    #------------------------------------------------------------

    Ahist = fit(Histogram, AS, 0.0 : 1.0e6 : 2.1e7, closed=:right);
    Bhist = fit(Histogram, BS, 0.0 : 1.0e6 : 2.1e7, closed=:right);
    Dhist = fit(Histogram, DS, 0.0 : 1.0e6 : 2.1e7, closed=:right);

    h = plot(size=(800,550), title="Openflight", xlabel=L"\rm{distance\ (m)}",
                                                 ylabel=L"\rm{edge\ probability}",
                                                 xlim=(3.5e+5, 2.5e+7),
                                                 ylim=(3.0e-6, 2.5e-2),
                                                 xscale=:log10,
                                                 yscale=:log10,
                                                 framestyle=:box,
                                                 grid="on");

    AprobL = linreg(log10.(0.5e6:1.0e6:1.15e7), log10.((Ahist.weights./Dhist.weights)[1:12]));
    BprobL = linreg(log10.(0.5e6:1.0e6:1.15e7), log10.((Bhist.weights./Dhist.weights)[1:12]));

    xrange = 0.5e6:0.1e6:1.15e7;
    AfitL = 10.^(AprobL[2] * log10.(xrange) + AprobL[1]);
    BfitL = 10.^(BprobL[2] * log10.(xrange) + BprobL[1]);

    plot!(h, xrange, AfitL, label="", color="red",  linestyle=:dot, linewidth=2.0);
    plot!(h, xrange, BfitL, label="", color="blue", linestyle=:dot, linewidth=2.0);

    scatter!(h, 0.5e6:1.0e6:1.35e7, (Ahist.weights./Dhist.weights)[1:14], label="original",  color="red",  ms=7);
    scatter!(h, 0.5e6:1.0e6:1.35e7, (Bhist.weights./Dhist.weights)[1:14], label="generated", color="blue", ms=7);

    savefig(h, "results/openflight_probE.pdf");

    return AS, BS, DS, h
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function hist_openflight()
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
    bt = BallTree(flipdim(coordinates,1), Haversine(6371e3));
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
            push!(dist_array, haversine(flipdim(id2lc[id1],1), flipdim(id2lc[id2],1), 6371e3));
            push!(rank_array, size(inrange(bt, flipdim(id2lc[id1],1), haversine(flipdim(id2lc[id1],1), flipdim(id2lc[id2],1), 6371e3)), 1));
            push!(rank_array, size(inrange(bt, flipdim(id2lc[id2],1), haversine(flipdim(id2lc[id1],1), flipdim(id2lc[id2],1), 6371e3)), 1));
        end
    end
    #--------------------------------

    return Array{Float64,1}(dist_array), Array{Int64,1}(rank_array)
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function hist_distance_rank(arr, b=0:50:6000)
    h = plot(rank_array,
             bins=b,
             legend=:topright,
             seriestype=:histogram,
             label="openflight data",
             title="histogram of connections w.r.t. rank distance",
             xlabel=L"$\min[\rm{rank}_{u}(v), \rm{rank}_{v}(u)]$",
             ylabel="counts")

    h = plot!(b[2:end], 26000 * b[2] ./ b[2:end], label=L"$1/\min[\rm{rank}_{u}(v), \rm{rank}_{v}(u)]$", size=(600,400));

    savefig(h, "results/air_rank_distance_hist.pdf")

    return h
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_openflight(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000)
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
    coordinates = [];
    coords = zeros(2,num_airports);
    for i in 1:num_airports
        push!(coordinates, id2lc[no2id[i]])
        coords[:,i] = flipdim(id2lc[no2id[i]],1)
    end
    #--------------------------------

    opt = Dict();
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;

    if (epsilon > 0)
        # D = Haversine_matrix(coordinates);
        # @time C, epsilon = StochasticCP.model_fit(A, D, epsilon; opt=opt);
        # B = StochasticCP.model_gen(C, D, epsilon);
        # C, epsilon = StochasticCP_SGD.model_fit(A, D, epsilon; opt=opt);
        @time C, epsilon = StochasticCP_FMM.model_fit(A, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        B = StochasticCP_FMM.model_gen(C, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        D = nothing;
        # B = StochasticCP.model_gen(C, D, epsilon);
    elseif (epsilon < 0)
        D = rank_distance_matrix(Haversine_matrix(coordinates));
        C, epsilon = StochasticCP.model_fit(A, D, -epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    else
        D = ones(A)-eye(A);
        C, epsilon = StochasticCP.model_fit(A, D, 1; opt=opt);
        B = StochasticCP.model_gen(C, D, 1);
    end

    return A, B, C, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_openflight(A, C, coords, option="degree", filename="output")
    h = plot(size=(800,450), title="Openflight",
                             xlabel=L"\rm{Longitude}(^\circ)",
                             ylabel=L"\rm{Latitude}(^\circ)",
                             framestyle=:box);

    plot_core_periphery(h, A, C, [flipdim(coord,1) for coord in coords], option;
                        plot_links=true,
                        distance="Haversine")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_brightkite(epsilon=1; ratio=1.0, thres=1.0e-6, max_num_step=1000)
    #--------------------------------
    # load airport data and location
    #--------------------------------
    brightkite_dat = readdlm("data/brightkite/Brightkite_totalCheckins.txt");
    num_checkins = size(brightkite_dat,1);

    #--------------------------------
    no2id = Dict{Int64, Int64}();
    id2no = Dict{Int64, Int64}();
    id2lc = Dict{Int64, Array{Float64,1}}();
    #--------------------------------
    # only keep the last check in location
    #--------------------------------
    num_people = 0;
    #--------------------------------
    for i in num_checkins:-1:1
        if (!(brightkite_dat[i,1] in keys(id2no)))
            num_people += 1;
            no2id[num_people] = brightkite_dat[i,1];
            id2no[brightkite_dat[i,1]] = num_people;
        end

        id2lc[brightkite_dat[i,1]] = brightkite_dat[i,3:4];
    end
    #--------------------------------

    #--------------------------------
    W = spzeros(num_people,num_people);
    #--------------------------------
    # the adjacency matrix
    #--------------------------------
    edges_dat = convert(Array{Int64,2}, readdlm("data/brightkite/Brightkite_edges.txt"));
    num_edges = size(edges_dat,1);
    for i in 1:num_edges
        id1 = edges_dat[i,1];
        id2 = edges_dat[i,2];
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
    # compute coordinates
    #--------------------------------
    coordinates = [];
    coords = zeros(2,num_people);
    for i in 1:num_people
        coord = id2lc[no2id[i]];
        coord = coord + rand(2);
        coord[1] = min(90, max(-90, coord[1]));
        coord[2] = coord[2] - floor((coord[2]+180.0) / 360.0) * 360.0;
        push!(coordinates, coord);
        coords[:,i] = flipdim(coord,1);
    end
    #--------------------------------

    opt = Dict();
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;

    if (epsilon > 0)
        @time C, epsilon = StochasticCP_FMM.model_fit(A, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        B = StochasticCP_FMM.model_gen(C, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        D = nothing;
    else
        error("option not supported.");
    end

    return A, B, C, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_brightkite(A, C, coords, option="degree", filename="output")
    h = plot(size=(800,450), title="Brightkite",
                             xlabel=L"\rm{Longitude}(^\circ)",
                             ylabel=L"\rm{Latitude}(^\circ)",
                             framestyle=:box);

    plot_core_periphery(h, A, C, [flipdim(coord,1) for coord in coords], option;
                        plot_links=false,
                        distance="Haversine")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------



#----------------------------------------------------------------
function test_mushroom(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000)
    #--------------------------------
    # load fungals data
    #--------------------------------
    data = MAT.matread("data/fungal_networks/Conductance/Ag_M_I+4R_U_N_42d_1.mat");
#   data = MAT.matread("data/fungal_networks/Conductance/Pv_M_5xI_U_N_35d_1.mat");
    coords = data["coordinates"]';
    A = spones(data["A"]);
    #--------------------------------

    coordinates = [[coords[1,i], coords[2,i]] for i in 1:size(coords,2)];

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;

    #--------------------------------
    if (epsilon > 0)
        D = Euclidean_matrix(coordinates);
        # C, epsilon = StochasticCP.model_fit(A, D, epsilon; opt=opt);
        # C, epsilon = StochasticCP_SGD.model_fit(A, D, epsilon; opt=opt);
        C, epsilon = StochasticCP_FMM.model_fit(A, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        # B = StochasticCP.model_gen(C, D, epsilon);
        B = StochasticCP_FMM.model_gen(C, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=Dict("ratio"=>0.0));
    elseif (epsilon < 0)
        D = rank_distance_matrix(Euclidean_matrix(coordinates));
        C, epsilon = StochasticCP.model_fit(A, D, -epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    else
        D = ones(A)-eye(A);
        C, epsilon = StochasticCP.model_fit(A, D, 1; opt=opt);
        B = StochasticCP.model_gen(C, D, 1);
    end

    return A, B, C, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_mushroom(A, C, coords, option="degree", filename="output")
    h = plot(size=(450,350), title="Mushroom",
                             xlabel=L"x",
                             ylabel=L"y",
                             framestyle=:box);

    plot_core_periphery(h, A, C, coords, option;
                        plot_links=true,
                        distance="Euclidean")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_celegans(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000)
    #--------------------------------
    # load fungals data
    #--------------------------------
    data = MAT.matread("data/celegans/celegans277.mat");
#   data = MAT.matread("data/fungal_networks/Conductance/Pv_M_5xI_U_N_35d_1.mat");
    coords = data["celegans277positions"]';
    A = spones(sparse(data["celegans277matrix"] + data["celegans277matrix"]'));
    #--------------------------------

    coordinates = [[coords[1,i], coords[2,i]] for i in 1:size(coords,2)];

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;

    #--------------------------------
    if (epsilon > 0)
        D = Euclidean_matrix(coordinates);
        C, epsilon = StochasticCP.model_fit(A, D, epsilon; opt=opt);
#       C, epsilon = StochasticCP_FMM.model_fit(A, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    elseif (epsilon < 0)
        D = rank_distance_matrix(Euclidean_matrix(coordinates));
        C, epsilon = StochasticCP.model_fit(A, D, -epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    else
        D = ones(A)-eye(A);
        C, epsilon = StochasticCP.model_fit(A, D, 1; opt=opt);
        B = StochasticCP.model_gen(C, D, 1);
    end

    return A, B, C, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_celegans(A, C, coords, option="degree", filename="output")
    h = plot(size=(450,350), title="Celegans",
                             xlabel=L"x",
                             ylabel=L"y",
                             framestyle=:box);

    plot_core_periphery(h, A, C, coords, option;
                        plot_links=true,
                        distance="Euclidean")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function intro_celegans(A, C, coords, option="community", filename="output")
    h = plot(size=(800,550), title="Celegans",
                             xlabel=L"x",
                             ylabel=L"y",
                             grid="off",
                             framestyle=:box);

    data = MAT.matread("data/celegans/celegans277.mat");
    com = data["celegans277communities"];

    @assert issymmetric(A);
    n = size(A, 1);
    d = vec(sum(A,1));

    color_set = [colorant"#ff0000", colorant"#00af00", colorant"#0000ff", colorant"#ffe119", colorant"#f58231",
                 colorant"#911eb4", colorant"#46f0f0", colorant"#fabebe", colorant"#008080", colorant"#e6beff"]

    if (option == "community")
        color = [color_set[com[i]] for i in 1:n];
    elseif (option == "core_periphery")
        #--------------------------------------------------------
        vtx_sorted = sortperm(C, rev=true);
        num_core = Int64(ceil(0.1335*n));
        Cmax = C[vtx_sorted[1]];
        Cmid = C[vtx_sorted[num_core]];
        Cmin = C[vtx_sorted[end]];
        #--------------------------------------------------------
        Rmap = Colors.linspace(colorant"#ffe6e6", colorant"#ff0000", 100);
        Gmap = Colors.linspace(colorant"#008000", colorant"#e6ffe6", 100);
        Bmap = Colors.linspace(colorant"#0000ff", colorant"#e6e6ff", 100);
        #--------------------------------------------------------
        color = [(i in vtx_sorted[1:num_core] ? Rmap[Int64(ceil((C[i]-Cmid)/(Cmax-Cmid) * 99)) + 1]
                                              : Bmap[Int64(ceil((C[i]-Cmin)/(Cmid-Cmin) * 99)) + 1]) for i in 1:n];
        #--------------------------------------------------------
    else
        error("option not supported.");
    end

    #------------------------------------------------------------
    if (length(coords[1]) == 2)
        #--------------------------------------------------------
        for i in 1:n
            for j in i+1:n
                #------------------------------------------------
                if (A[i,j] != 0)
                    plot!(h, [coords[i][1], coords[j][1]],
                             [coords[i][2], coords[j][2]],
                             legend=false,
                             color="grey",
                             linewidth=0.1,
                             alpha=0.1);
                end
                #------------------------------------------------
            end
        end
        #--------------------------------------------------------
    end
    #------------------------------------------------------------
    scatter!(h, [coord[1] for coord in coords], [coord[2] for coord in coords], ms=sqrt.(d)*1.8, c=color, alpha=1.00);
    #----------------------------------------------------------------

    savefig(h, "results/" * filename * ".svg");

    if (option == "community")
        od = [i for j in 1:10 for i in shuffle(1:n) if com[i] == j];
    elseif (option == "core_periphery")
        od = sortperm(C, rev=true);
    end

    #----------------------------------------------------------------
    R = zeros(n,n);
    #----------------------------------------------------------------
    for i in 1:n
        for j in 1:n
            R[i,j] = A[od[i], od[j]];
        end
    end
    #----------------------------------------------------------------

    return h, R;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function Karate(thres=1.0e-6, max_num_step=100)
    #--------------------------------
    # load Karate club data
    #--------------------------------
    data = MAT.matread("data/karate/karate.mat");
    #--------------------------------
    A = data["A"];
    pos = data["pos"];
    #--------------------------------
    @assert issymmetric(A);
    #--------------------------------s
    n = size(A,1);
    #--------------------------------

    #--------------------------------
    coordinates = [];
    coords = zeros(2,n);
    for i in 1:n
        push!(coordinates, pos[i,:])
        coords[:,i] = pos[i,:]
    end
    #--------------------------------

    #--------------------------------
    D = ones(A)-eye(A);
    C, epsilon = StochasticCP.model_fit(A, D, 1; opt=Dict("thres"=>1.0e-6, "max_num_step"=>100));
    B = StochasticCP.model_gen(C, D, 1);
    #--------------------------------

    return A, B, C, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function intro_Karate(A, C, coords, option="community", filename="output")
    h = plot(size=(800,550), title="Karate",
                             xlabel=L"x",
                             ylabel=L"y",
                             grid="off",
                             framestyle=:box);

    data = MAT.matread("data/karate/karate.mat");
    com = data["com"];

    @assert issymmetric(A);
    n = size(A, 1);

    if (option == "community")
        color = [(i in com[2] ? colorant"#FF3333" : colorant"#33FFFF") for i in 1:n];
    elseif (option == "core_periphery")
        color = [(i in sortperm(C, rev=true)[1:Int64(ceil(0.55*n))] ? colorant"#FF3333" : colorant"#33FFFF") for i in 1:n];
    else
        error("option not supported.");
    end

    #------------------------------------------------------------
    if (length(coords[1]) == 2)
        #--------------------------------------------------------
        for i in 1:n
            for j in i+1:n
                #------------------------------------------------
                if (A[i,j] != 0)
                    plot!(h, [coords[i][1], coords[j][1]],
                             [coords[i][2], coords[j][2]],
                             legend=false,
                             color="grey",
                             linewidth=0.2,
                             alpha=1.0);
                end
                #------------------------------------------------
            end
        end
        #--------------------------------------------------------
    end
    #------------------------------------------------------------
    scatter!(h, [coord[1] for coord in coords], [coord[2] for coord in coords], ms=ones(n)*20, c=color, alpha=1.00);
    #----------------------------------------------------------------
    annotate!([(coords[i][1], coords[i][2], string(i), 15) for i in 1:n]);
    #----------------------------------------------------------------

    savefig(h, "results/" * filename * ".svg");

    if (option == "community")
        od = vcat(vec(com[2]), vec(com[1]));
    elseif (option == "core_periphery")
        od = sortperm(C, rev=true);
    end

    #----------------------------------------------------------------
    R = zeros(n,n);
    #----------------------------------------------------------------
    for i in 1:n
        for j in 1:n
            R[i,j] = A[od[i], od[j]];
        end
    end
    #----------------------------------------------------------------

    return h, R;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_facebook(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000)
    #--------------------------------
    # load facebook100 data
    #--------------------------------
#   data = MAT.matread("data/facebook100/Cornell5.mat");
    data = MAT.matread("data/facebook100/Caltech36.mat");
    A = spones(data["A"]);
    @assert issymmetric(A);
    #--------------------------------
    n = size(A,1);
    coords = vcat(convert(Array{Float64,2}, reshape(collect(1:n), (1,n))), data["local_info"]');
    #--------------------------------

    coordinates = [[coords[i,j] for i in 1:8] for j in 1:size(coords,2)];

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;

    #--------------------------------
    if (epsilon > 0)
        D = Hamming_matrix(coordinates);
        C, epsilon = StochasticCP.model_fit(A, D, epsilon; opt=opt);
        # C, epsilon = StochasticCP_SGD.model_fit(A, D, epsilon; opt=opt);
        # C, epsilon = StochasticCP_FMM.model_fit(A, coords, Hamming_CoM2, Hamming(), epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    elseif (epsilon < 0)
        D = rank_distance_matrix(Hamming_matrix(coordinates));
        C, epsilon = StochasticCP.model_fit(A, D, -epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
    else
        D = ones(A)-eye(A);
        C, epsilon = StochasticCP.model_fit(A, D, 1; opt=opt);
        B = StochasticCP.model_gen(C, D, 1);
    end

    return A, B, C, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_facebook(A, C, coords, option="degree", filename="output")
    h = plot(size=(600,600), title="Facebook",
                             xlabel=L"status",
                             ylabel=L"year");

    plot_core_periphery(h, A, C, [[(coord[2] >=    0 ? coord[2] :    0) + (rand()-0.5)*0.3,
                                   (coord[7] >= 1999 ? coord[7] : 1999) + (rand()-0.5)*0.3] for coord in coords], option;
                        plot_links=true,
                        distance="Euclidean")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_algo(sigma, num_vertices)
    h = plot(size=(600,600), title="",
                             xlabel=L"x",
                             ylabel=L"y",
                             xlim=(0.0, 1.0),
                             ylim=(0.0, 1.0),
                             grid="off",
                             framestyle=:none);

    centers = [[0.10, 0.65], [0.30, 0.65], [0.60, 0.65], [0.80, 0.65],
               [0.10, 0.35], [0.30, 0.35], [0.60, 0.35], [0.80, 0.35]];
    ctypes = [colorant"#00a0ff", colorant"#00e0ff", colorant"#ffa000", colorant"#ffe000",
              colorant"#0060ff", colorant"#0020ff", colorant"#ff6000", colorant"#ff2000"];

    coords = [];
    colors = [];

    for i in 1:8
        for j in 1:num_vertices[i]
            push!(coords, centers[i]+randn(2)*sigma);
            push!(colors, ctypes[i]);
        end
    end

    n = size(coords,1);

    C = ones(n) * (-3.3);
    D = Euclidean_matrix(coords);
    A = StochasticCP.model_gen(C, D, 2);

    #------------------------------------------------------------
    if (length(coords[1]) == 2)
        #--------------------------------------------------------
        for i in 1:n
            for j in i+1:n
                #------------------------------------------------
                if ((A[i,j] != 0) && (colors[i] != colors[j]))
                    if ((colors[i] == ctypes[1] && colors[j] == ctypes[2]) ||
                        (colors[i] == ctypes[3] && colors[j] == ctypes[4]) ||
                        (colors[i] == ctypes[5] && colors[j] == ctypes[6]) ||
                        (colors[i] == ctypes[7] && colors[j] == ctypes[8]))
                        plot!(h, [coords[i][1], coords[j][1]],
                                 [coords[i][2], coords[j][2]],
                                 legend=false,
                                 color="black",
                                 linewidth=1.0,
                                 linestyle=:dash,
                                 alpha=0.3);
                    else
                        plot!(h, [coords[i][1], coords[j][1]],
                                 [coords[i][2], coords[j][2]],
                                 legend=false,
                                 color="black",
                                 linewidth=0.5,
                                 linestyle=:solid,
                                 alpha=0.15);
                    end
                end
                #------------------------------------------------
            end
        end
        #--------------------------------------------------------
    end
    #------------------------------------------------------------
    scatter!(h, [coord[1] for coord in coords], [coord[2] for coord in coords], ms=6, c=colors, alpha=1.00);
    #----------------------------------------------------------------

    savefig(h, "results/algo_network.svg");

    return h;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function check(A, C, D, coordinates, metric, CoM2, epsilon, ratio)
    coords = flipdim([coordinates[i][j] for i in 1:size(coordinates,1), j in 1:2]',1);
    bt = BallTree(coords, metric, leafsize=1);
    dist = Dict{Int64,Array{Float64,1}}(i => vec(D[:,i]) for i in 1:length(C));

    #-----------------------------------------------------------------------------
    # \sum_{ij in E} -log_Dij
    #-----------------------------------------------------------------------------
    I,J,V = findnz(A);
    #-----------------------------------------------------------------------------
    sum_logD_inE = 0.0
    #-----------------------------------------------------------------------------
    for (i,j) in zip(I,J)
        #---------------------------------------------------------------------
        if (i < j)
            sum_logD_inE += log(Distances.evaluate(metric, coords[:,i], coords[:,j]))
        end
        #---------------------------------------------------------------------
    end
    #-----------------------------------------------------------------------------

    omega_real = StochasticCP.omega(A, C, D, epsilon);
    omega = StochasticCP_FMM.omega!(C, coords, CoM2, dist, epsilon, bt, ratio, A, sum_logD_inE);

    epd_real = vec(sum(StochasticCP.probability_matrix(C, D, epsilon), 1));
    epd, srd, fmm_tree = StochasticCP_FMM.epd_and_srd!(C, coords, CoM2, dist, epsilon, bt, ratio);

    order = sortperm(C, rev=false);
    h = plot(epd_real[order]);
    plot!(h, epd[order]);
    plot!(h, epd[order] - epd_real[order]);


    return h, fmm_tree, omega_real, omega;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function timeit(n, metric, CoM2, epsilon)
    #------------------------------------------------------------
    coords = rand(2,n);
    #------------------------------------------------------------
    bt = BallTree(coords, metric, leafsize=1);
    #------------------------------------------------------------

    #------------------------------------------------------------
    II = Int64[];
    JJ = Int64[];
    VV = Float64[];
    #------------------------------------------------------------
    for counter in 1:10*n
        id_1 = rand(1:n);
        id_2 = rand(1:n);
        if (id_1 != id_2)
            if (id_1 > id_2)
                id_1, id_2 = id_2, id_1
            end

            push!(II, id_1);
            push!(JJ, id_2);
            push!(VV, 1.0);
        end
    end
    #------------------------------------------------------------
    A = sparse(II,JJ,VV, n,n, max);
    #------------------------------------------------------------
    A = A + A';
    #------------------------------------------------------------

    C = ones(n) * ((-0.5) * log(sqrt(3) * (n/5 - 1)));

    println("start");

    if (n <= 1.0e4)
        #------------------------------------------------------------
        D = zeros(n,n);
        #------------------------------------------------------------
        for j in 1:n
            for i in j+1:n
                D[i,j] = Distances.evaluate(metric, coords[:,i], coords[:,j]);
            end
        end
        #------------------------------------------------------------
        D = D + D';
        #------------------------------------------------------------
        @time (omega = StochasticCP.omega(A, C, D, epsilon);)
#       @time (epd_real = vec(sum(StochasticCP.probability_matrix(C, D, epsilon), 1)); srd = StochasticCP.sum_rho_logD(C,D,epsilon);)
#       @time (B_ori = StochasticCP.model_gen(C, D, epsilon));
    end

    @time (omega = StochasticCP_FMM.omega!(C, coords, CoM2, Dict(), epsilon, bt, 0.0, A, 0.0));
#   @time (epd, srd, fmm_tree = StochasticCP_FMM.epd_and_srd!(C, coords, CoM2, Dict(), epsilon, bt, 0.0));
#   @time (B_fmm = StochasticCP_FMM.model_gen(C, coords, CoM2, metric, epsilon; opt = Dict("ratio"=>0.0)));

#   if (n <= 1.0e4)
#       return countnz(B_ori)/n^2, countnz(B_fmm)/n^2;
#   else
#       return countnz(B_fmm)/n^2;
#   end
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_timings()
    #------------------------------------------------------------
    size = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6];
    #------------------------------------------------------------
    ori_omega = [0.002032, 0.116223, 17.009791, 1700.000000, 170000.000000];
    fmm_omega = [0.000610, 0.008427,  0.087300,    1.203448,     12.367116];
    #------------------------------------------------------------
    ori_deriv = [0.001447, 0.111002, 26.671128, 2600.000000, 260000.000000];
    fmm_deriv = [0.001503, 0.023390,  0.286124,    3.228055,     39.302156];
    #------------------------------------------------------------
    ori_gener = [0.000786, 0.039290,  4.137897,  413.000000, 41300.000000];
    fmm_gener = [0.003978, 0.103288,  1.768498,   22.576165,   256.542071];
    #------------------------------------------------------------

    h = plot(size=(600,450), title="Timings", xlabel="number of vertices",
                                              ylabel="time per function call (sec)",
                                              xlim=(10^(+1.7), 10^(+6.3)),
                                              ylim=(10^(-3.7), 10^(+2.7)),
                                              xscale=:log10,
                                              yscale=:log10,
                                              framestyle=:box,
                                              grid="on");

    exp_omega = (size .* log.(size))    * (fmm_omega[1] / (size[1] * log(size[1])  ));
    exp_deriv = (size .* log.(size))    * (fmm_deriv[1] / (size[1] * log(size[1])  ));
    exp_gener = (size .* log.(size).^2) * (fmm_gener[1] / (size[1] * log(size[1])^2));

    scatter!(h, size, fmm_omega, label="objective function", color="red", ms=8);
    plot!(h, size, exp_omega, label=L"\mathcal{O}\left(|V| \cdot \log |V|\right)", color="red", linestyle=:dash, linewidth=2.0);
    scatter!(h, size, fmm_deriv, label="derivatives", color="blue", ms=8);
    plot!(h, size, exp_deriv, label=L"\mathcal{O}\left(|V| \cdot \log |V|\right)", color="blue", linestyle=:dash, linewidth=2.0);
    scatter!(h, size, fmm_gener, label="generate network", color="green", ms=8);
    plot!(h, size, exp_gener, label=L"\mathcal{O}\left(|V| \cdot (\log |V|)^2\right)", color="green", linestyle=:dash, linewidth=2.0);

    scatter!(h, size, fmm_omega, label="", color="red",   ms=8);
    scatter!(h, size, fmm_deriv, label="", color="blue",  ms=8);
    scatter!(h, size, fmm_gener, label="", color="green", ms=8);

    savefig(h, "results/fmm_timings.pdf");

    return h
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_ring(n, m, beta=0.0; epsilon=1.0, ratio=1.0, thres=1.0e-6, max_num_step=1000)
        #--------------------------------------------------------
        dis(i,j) = min(max(i,j) - min(i,j), min(i,j)+n - max(i,j));
        rd(i) = mod(i-1+n,n) + 1;

        A = spzeros(n,n);
        for i in 1:n
            for j in 1:m
                if (rand() < beta)
                    k = 0;
                    while ((k = rand(1:n)) == i)
                        # do nothing
                    end

                    A[i,k] = 1;
                    A[k,i] = 1;
                else
                    A[i,rd(i-j)] = 1;
                    A[rd(i-j),i] = 1;
                end

                if (rand() < beta)
                    k = 0;
                    while ((k = rand(1:n)) == i)
                        # do nothing
                    end

                    A[i,k] = 1;
                    A[k,i] = 1;
                else
                    A[i,rd(i+j)] = 1;
                    A[rd(i+j),i] = 1;
                end
            end
        end

        D = zeros(n,n);
        for i in 1:n
            for j in 1:n
                D[i,j] = dis(i,j);
            end
        end

        opt = Dict();
        opt["ratio"] = ratio;
        opt["thres"] = thres;
        opt["max_num_step"] = max_num_step;

        C, epsilon = StochasticCP.model_fit(A, D, epsilon; opt=opt);
        B = StochasticCP.model_gen(C, D, epsilon);
        #--------------------------------------------------------

        return A, B, C, epsilon
end
#----------------------------------------------------------------
