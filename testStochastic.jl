using StatsBase;
using MAT;
using Colors;
using Plots;
using LaTeXStrings;
using MatrixNetworks;
using Dierckx;
using Distances;
using Distributions;
using NearestNeighbors;
using SCP;
using SCP_SGD;
using SCP_FMM;
using NLsolve;
using LightGraphs;
using TikzGraphs;
using TikzPictures;
using DecisionTree;
using ScikitLearn;
using ScikitLearn.CrossValidation: cross_val_score;
@sk_import linear_model: LogisticRegression;

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
        order = sortperm(D[:,j]);
        for i in 2:n
            R[order[i],j] = i-1;
        end
    end

    R = min.(R,R');

    return R;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_core_periphery(h, A, theta, coords, option="degree";
                             plot_links=false,
                             distance="Euclidean")
    @assert issymmetric(A);
    n = size(A,1);

    d = vec(sum(A,1));

    color = [(i in sortperm(theta)[end-Int64(ceil(0.10*n)):end] ? colorant"orange" : colorant"blue") for i in 1:n];

    if (option == "degree")
        println("option: degree")
        order = sortperm(d);
        ms = (sqrt.(d) / sqrt(maximum(d))) * 3.6 + 1.0;
    elseif (option == "core_score")
        println("option: core_score")
        order = sortperm(theta);
        rk = sortperm(sortperm(theta))
        ms = (rk/n).^20 * 6.0 + 1.5;
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
                        Plots.plot!(h, [coords[i][1], coords[j][1]],
                                 [coords[i][2], coords[j][2]],
                                 legend=false,
                                 color="gray",
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
                            Plots.plot!(h, [coords[i][1], coords[j][1]],
                                     [coords[i][2], coords[j][2]],
                                     legend=false,
                                     color="gray",
                                     linewidth=0.10,
                                     alpha=0.10);
                        else
                            min_id = coords[i][1] <= coords[j][1] ? i : j;
                            max_id = coords[i][1] >  coords[j][1] ? i : j;

                            lat_c  = ((coords[min_id][2] - coords[max_id][2]) / ((coords[min_id][1] + 360) - coords[max_id][1])) * (180 - coords[max_id][1]) + coords[max_id][2]

                            Plots.plot!(h, [-180.0, coords[min_id][1]],
                                     [lat_c,  coords[min_id][2]],
                                     legend=false,
                                     color="gray",
                                     linewidth=0.10,
                                     alpha=0.10);

                            Plots.plot!(h, [coords[max_id][1], 180.0],
                                     [coords[max_id][2], lat_c],
                                     legend=false,
                                     color="gray",
                                     linewidth=0.10,
                                     alpha=0.10);
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
    Plots.scatter!(h, [coord[1] for coord in coords[order]], [coord[2] for coord in coords[order]], ms=ms[order], c=color[order], alpha=1.00,
                                                                                                                                  label="",
                                                                                                                                  markerstrokealpha=0.1,
                                                                                                                                  markerstrokewidth=0.0,
                                                                                                                                  markerstrokecolor="black");
    #----------------------------------------------------------------
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_underground(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=10000, opt_epsilon=true)
    dat = MAT.matread("data/london_underground/london_underground_clean_traffic.mat");

    W = [Int(sum(list .!= 0)) for list in dat["Labelled_Network"]];
    A = spones(sparse(W));

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = 2.0;
    opt["delta_2"] = 0.2;

    if (epsilon > 0)
        D = Haversine_matrix(dat["Tube_Locations"]);
        theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
        B = SCP.model_gen(theta, D, epsilon);
    elseif (epsilon < 0)
        D = rank_distance_matrix(Haversine_matrix(dat["Tube_Locations"]));
        theta, epsilon = SCP.model_fit(A, D, -epsilon; opt=opt);
        B = SCP.model_gen(theta, D, epsilon);
    else
        D = ones(A)-eye(A);
        theta, epsilon = SCP.model_fit(A, D, 1; opt=opt);
        B = SCP.model_gen(theta, D, 1);
    end

    return A, B, theta, D, dat["Tube_Locations"], epsilon;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function plot_underground(A, theta, coords, option="degree", filename="output")
    h = Plots.plot(size=(570,450), title="London Underground",
                             xlabel=L"\rm{Longitude}(^\circ)",
                             ylabel=L"\rm{Latitude}(^\circ)",
                             framestyle=:box,
                             grid="off");

    plot_core_periphery(h, A, theta, [flipdim(coord,1) for coord in coords], option;
                        plot_links=true,
                        distance="Haversine")

    Plots.savefig(h, "results/" * filename * ".svg");
    return h;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function analyze_underground(A,theta)
    degree = vec(sum(A,1));

    dat = MAT.matread("data/london_underground/london_underground_clean_traffic.mat");

    # random forest prediction
    base_vec = degree;
    theta_vec = theta;

    labels = convert(Array{Float64,1}, dat["Traffic"]);
    features1  = reshape(base_vec, :, 1);
    features2  = reshape(theta_vec, :, 1);
    features12 = convert(Array{Float64,2}, reshape([base_vec; theta_vec], :, 2));
    r1  = nfoldCV_forest(labels, features1,  1, 100, 3, 5, 0.7);
    r2  = nfoldCV_forest(labels, features2,  1, 100, 3, 5, 0.7);
    r12 = nfoldCV_forest(labels, features12, 2, 100, 3, 5, 0.7);

    println("\n\n\n(r1, r2, r12) = (", mean(r1), ", ", mean(r2), ", ", mean(r12), ")");

    return labels, features1, features2;
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

    Ahist = StatsBase.fit(Histogram, AS, 0.0 : 1.0e6 : 2.1e7, closed=:right);
    Bhist = StatsBase.fit(Histogram, BS, 0.0 : 1.0e6 : 2.1e7, closed=:right);
    Dhist = StatsBase.fit(Histogram, DS, 0.0 : 1.0e6 : 2.1e7, closed=:right);

    h = Plots.plot(size=(800,550), title="Openflight", xlabel=L"\rm{distance\ (m)}",
                                                 ylabel=L"\rm{edge\ probability}",
                                                 xlim=(3.5e+5, 2.5e+7),
                                                 ylim=(3.0e-6, 2.5e-2),
                                                 xscale=:log10,
                                                 yscale=:log10,
                                                 framestyle=:box,
                                                 grid="off");

    AprobL = linreg(log10.(0.5e6:1.0e6:1.15e7), log10.((Ahist.weights./Dhist.weights)[1:12]));
    BprobL = linreg(log10.(0.5e6:1.0e6:1.15e7), log10.((Bhist.weights./Dhist.weights)[1:12]));

    xrange = 0.5e6:0.1e6:1.15e7;
    AfitL = 10.^(AprobL[2] * log10.(xrange) + AprobL[1]);
    BfitL = 10.^(BprobL[2] * log10.(xrange) + BprobL[1]);

    Plots.plot!(h, xrange, AfitL, label="", color="red",  linestyle=:dot, linewidth=2.0);
    Plots.plot!(h, xrange, BfitL, label="", color="blue", linestyle=:dot, linewidth=2.0);

    Plots.scatter!(h, 0.5e6:1.0e6:1.35e7, (Ahist.weights./Dhist.weights)[1:14], label="original",  color="red",  ms=7);
    Plots.scatter!(h, 0.5e6:1.0e6:1.35e7, (Bhist.weights./Dhist.weights)[1:14], label="generated", color="blue", ms=7);

    Plots.savefig(h, "results/openflight_probE.pdf");

    return AS, BS, DS, h
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function celegans_gen_analysis(A, BB_nev, BB_fmm, D)
    @assert issymmetric(A);
    @assert issymmetric(D);

    #------------------------------------------------------------
    degrees_ori = vec(sum(A,1));
    degrees_nev = vec(sum(mean(BB_nev),1));
    degrees_fmm = vec(sum(mean(BB_fmm),1));
    #------------------------------------------------------------
    n = size(A,1);
    #------------------------------------------------------------

    #------------------------------------------------------------
    h1 = Plots.plot(size=(240,220), title="",
                              xlabel="degrees (original)",
                              ylabel="degrees (generated)",
                              xlim=(-2, 80),
                              ylim=(-2, 80),
                              grid="off",
                              framestyle=:box,
                              legend=:topleft);
    Plots.plot!(h1, -2:80, -2:80, color="red", label="ideal");
    degrees_original  = vcat(degrees_ori, degrees_ori);
    degrees_generated = vcat(degrees_nev, degrees_fmm);
    color = [i <= n ? colorant"blue" : colorant"orange" for i in 1:2*n];
    order = randperm(2*n);
    #------------------------------------------------------------
    Plots.scatter!(h1, [], [], color="blue",   label="naive", markerstrokewidth=0.1, markerstrokecolor="blue",   markerstrokealpha=1.0, markersize=3.0, markeralpha=0.6);
    Plots.scatter!(h1, [], [], color="orange", label="FMM",   markerstrokewidth=0.1, markerstrokecolor="orange", markerstrokealpha=1.0, markersize=3.0, markeralpha=0.6);
    Plots.scatter!(h1, degrees_original[order], degrees_generated[order], color=color[order], label="", markerstrokewidth=0.1, markerstrokecolor=color[order], markerstrokealpha=1.0, markersize=3.0, markeralpha=0.6);
    #------------------------------------------------------------

    #------------------------------------------------------------
    AS     = Array{Float64,1}();
    BS_nev = Array{Float64,1}();
    BS_fmm = Array{Float64,1}();
    DS     = Array{Float64,1}();
    #------------------------------------------------------------
    for i in 1:n
        for j in i+1:n
            #----------------------------------------------------
            if (A[i,j] == 1)
                push!(AS, D[i,j]);
            end
            #----------------------------------------------------
            for k in 1:length(BB_nev)
                if (BB_nev[k][i,j] == 1)
                    push!(BS_nev, D[i,j]);
                end
            end
            #----------------------------------------------------
            for k in 1:length(BB_fmm)
                if (BB_fmm[k][i,j] == 1)
                    push!(BS_fmm, D[i,j]);
                end
            end
            #----------------------------------------------------
        end
    end
    #------------------------------------------------------------

    accumulated_ori = Array{Float64,1}();
    accumulated_nev = Array{Float64,1}();
    accumulated_fmm = Array{Float64,1}();

    thresholds = 0.00:0.01:1.35;

    for thres in thresholds
        push!(accumulated_ori, sum(AS     .< thres));
        push!(accumulated_nev, sum(BS_nev .< thres)/length(BB_nev));
        push!(accumulated_fmm, sum(BS_fmm .< thres)/length(BB_fmm));
    end

    #------------------------------------------------------------
    h2 = Plots.plot(size=(250,220), title="",
                              xlabel="distance threshold (mm)",
                              ylabel="number of edges",
                              xlim=(0.00,1.35),
                              ylim=(0, 2000),
                              xticks=[0.0, 0.3, 0.6, 0.9, 1.2],
                              grid="off",
                              framestyle=:box,
                              legend=:bottomright);
    #------------------------------------------------------------
    Plots.plot!(h2, thresholds, accumulated_ori, linewidth=3.5, linestyle=:solid, color="grey",   label="original");
    Plots.plot!(h2, thresholds, accumulated_nev, linewidth=2.0, linestyle=:solid, color="blue",   label="naive");
    Plots.plot!(h2, thresholds, accumulated_fmm, linewidth=1.5, linestyle=:solid, color="orange", label="FMM");

#     Ahist = fit(Histogram, AS, 0.0 : 1.0e6 : 2.1e7, closed=:right);
#     Bhist = fit(Histogram, BS, 0.0 : 1.0e6 : 2.1e7, closed=:right);
#     Dhist = fit(Histogram, DS, 0.0 : 1.0e6 : 2.1e7, closed=:right);
#
#     h = Plots.plot(size=(800,550), title="Openflight", xlabel=L"\rm{distance\ (m)}",
#                                                  ylabel=L"\rm{edge\ probability}",
#                                                  xlim=(3.5e+5, 2.5e+7),
#                                                  ylim=(3.0e-6, 2.5e-2),
#                                                  xscale=:log10,
#                                                  yscale=:log10,
#                                                  framestyle=:box,
#                                                  grid="on");
#
#     AprobL = linreg(log10.(0.5e6:1.0e6:1.15e7), log10.((Ahist.weights./Dhist.weights)[1:12]));
#     BprobL = linreg(log10.(0.5e6:1.0e6:1.15e7), log10.((Bhist.weights./Dhist.weights)[1:12]));
#
#     xrange = 0.5e6:0.1e6:1.15e7;
#     AfitL = 10.^(AprobL[2] * log10.(xrange) + AprobL[1]);
#     BfitL = 10.^(BprobL[2] * log10.(xrange) + BprobL[1]);
#
#     Plots.plot!(h, xrange, AfitL, label="", color="red",  linestyle=:dot, linewidth=2.0);
#     Plots.plot!(h, xrange, BfitL, label="", color="blue", linestyle=:dot, linewidth=2.0);
#
#     Plots.scatter!(h, 0.5e6:1.0e6:1.35e7, (Ahist.weights./Dhist.weights)[1:14], label="original",  color="red",  ms=7);
#     Plots.scatter!(h, 0.5e6:1.0e6:1.35e7, (Bhist.weights./Dhist.weights)[1:14], label="generated", color="blue", ms=7);
#
#     Plots.savefig(h, "results/openflight_probE.pdf");

    Plots.savefig(h1, "results/celegans_degrees.svg");
    Plots.savefig(h2, "results/celegans_hist_distance.svg");

    return AS, BS_nev, BS_fmm, DS, h1, h2
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
    h = Plots.plot(rank_array,
             bins=b,
             legend=:topright,
             seriestype=:histogram,
             label="openflight data",
             title="histogram of connections w.r.t. rank distance",
             xlabel=L"$\min[\rm{rank}_{u}(v), \rm{rank}_{v}(u)]$",
             ylabel="counts")

    h = Plots.plot!(b[2:end], 26000 * b[2] ./ b[2:end], label=L"$1/\min[\rm{rank}_{u}(v), \rm{rank}_{v}(u)]$", size=(600,400));

    Plots.savefig(h, "results/air_rank_distance_hist.pdf")

    return h
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_openflight(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true, delta_1=2.0, delta_2=0.2)
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
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = delta_1;
    opt["delta_2"] = delta_2;

    if (epsilon > 0)
        # D = Haversine_matrix(coordinates);
        # @time theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
        # B = SCP.model_gen(theta, D, epsilon);
        # theta, epsilon = SCP_SGD.model_fit(A, D, epsilon; opt=opt);
        t1 = time_ns()
        @time theta, epsilon, optim = SCP_FMM.model_fit(A, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        t2 = time_ns()

        B = SCP_FMM.model_gen(theta, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        D = Haversine_matrix(coordinates);
        # B = SCP.model_gen(theta, D, epsilon);
    elseif (epsilon < 0)
        D = rank_distance_matrix(Haversine_matrix(coordinates));
        theta, epsilon = SCP.model_fit(A, D, -epsilon; opt=opt);
        B = SCP.model_gen(theta, D, epsilon);
    else
        D = ones(A)-eye(A);
        theta, epsilon = SCP.model_fit(A, D, 1; opt=opt);
        B = SCP.model_gen(theta, D, 1);
    end

    return A, B, theta, D, coordinates, epsilon, (t2-t1)/1.0e9/optim.f_calls;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function tradeoff_openflight(delta1_array, delta2_array)
    theta0 = MAT.matread("results/openflight_distanceopt.mat")["C"];

    delta1_time = Vector{Float64}();
    delta1_rmse = Vector{Float64}();
    delta1_corr = Vector{Float64}();
    delta1_thta = Vector{Vector{Float64}}();
    delta1_epsl = Vector{Float64}();
    for delta1 in delta1_array
        A,B,theta,D,coordinates,epsilon,t = test_openflight(2.3; ratio=0.00, max_num_step=1000, opt_epsilon=true, delta_1=delta1);
        push!(delta1_time, t);
        push!(delta1_rmse, (sum((theta-theta0).^2)/length(theta0))^0.5);
        push!(delta1_corr, cor(theta,theta0));
        push!(delta1_thta, theta);
        push!(delta1_epsl, epsilon);
    end

    delta2_time = Vector{Float64}();
    delta2_rmse = Vector{Float64}();
    delta2_corr = Vector{Float64}();
    delta2_thta = Vector{Vector{Float64}}();
    delta2_epsl = Vector{Float64}();
    for delta2 in delta2_array
        A,B,theta,D,coordinates,epsilon,t = test_openflight(1.0; ratio=0.00, max_num_step=1000, opt_epsilon=true, delta_2=delta2);
        push!(delta2_time, t);
        push!(delta2_rmse, (sum((theta-theta0).^2)/length(theta0))^0.5);
        push!(delta2_corr, cor(theta,theta0));
        push!(delta2_thta, theta);
        push!(delta2_epsl, epsilon);
    end

    return delta1_time, delta1_rmse, delta1_corr, delta1_thta, delta1_epsl, delta2_time, delta2_rmse, delta2_corr, delta2_thta, delta2_epsl;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_openflight(A, theta, coords, option="degree", filename="output")
#   h = Plots.plot(size=(570,350), title="Openflight",
#                            xlabel=L"\rm{Longitude}(^\circ)",
#                            ylabel=L"\rm{Latitude}(^\circ)",
#                            xticks=[],
#                            yticks=[]
#                            framestyle=:none,
#                            grid="off");

    h = Plots.plot(size=(600,300), title="",
                             xlabel="",
                             ylabel="",
                             xticks=[],
                             yticks=[],
                             xlim = [-180, 180],
                             ylim = [-70, 80],
                             framestyle=:box,
                             grid="off");

    plot_core_periphery(h, A, theta, [flipdim(coord,1) for coord in coords], option;
                        plot_links=true,
                        distance="Haversine")

    Plots.savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function analyze_openflight(A,theta)
    #--------------------------------------------------
    lg = Graph(A);
    #--------------------------------------------------
    degree = vec(sum(A,1));
    btcetr = LightGraphs.betweenness_centrality(lg);
    clcetr = LightGraphs.closeness_centrality(lg);
    evcetr = LightGraphs.eigenvector_centrality(lg);
    pagerk = LightGraphs.pagerank(lg);
    #--------------------------------------------------

    dat_world = readcsv("data/open_airlines/airports.dat");
    dat_US = readcsv("data/open_airlines/enplanements.csv");
    dat_US_code = dat_US[:, 4];
    dat_US_epmt = dat_US[:,10];

    code2id = Dict(dat_world[i,5] => i for i in 1:7184);

    indices  = [i for i in 1:length(dat_US_code) if ((dat_US_code[i] in keys(code2id)) && (dat_US_code[i] != "") && (degree[code2id[dat_US_code[i]]] != 0))];
    code_vec = [dat_US_code[id] for id in indices];
    empt_vec = [parse(Int64, replace(dat_US_epmt[id], ",", "")) for id in indices];
    base_vec = [clcetr[code2id[dat_US_code[id]]] for id in indices];
    theta_vec = [theta[code2id[dat_US_code[id]]] for id in indices];

    labels = convert(Array{Float64,1}, empt_vec);
    features01 = reshape(base_vec, :, 1);
    features10 = reshape(theta_vec, :, 1);
    features11 = convert(Array{Float64,2}, reshape([base_vec; theta_vec], :, 2));

    #--------------------------------------------------
    # manually test the correlation coefficient
    #--------------------------------------------------
    r  = Vector{Float64}();
    #--------------------------------------------------
    ratio = 0.8;
    #--------------------------------------------------
    permlist = randperm(length(labels));
    tranlist = permlist[1:Int64(floor(length(labels) * ratio))];
    testlist = permlist[Int64(floor(length(labels) * ratio))+1:end];
    #--------------------------------------------------
    model01 = build_tree(labels[tranlist], features01[tranlist,:], 0, 0, 3); model01 = prune_tree(model01, 0.9);
    model10 = build_tree(labels[tranlist], features10[tranlist,:], 0, 0, 3); model10 = prune_tree(model10, 0.9);
    model11 = build_tree(labels[tranlist], features11[tranlist,:], 0, 0, 3); model11 = prune_tree(model11, 0.9);
    #--------------------------------------------------
    labels_test = apply_tree(model10, features10[testlist,:]);
    #--------------------------------------------------
    push!(r, DecisionTree.R2(labels[testlist], labels_test));
    #--------------------------------------------------

    #--------------------------------------------------
    # use decision tree to predict empt
    #--------------------------------------------------
    r01 = Vector{Float64}();
    r10 = Vector{Float64}();
    r11 = Vector{Float64}();
    #--------------------------------------------------
    for itr in 1:30
        r01 = vcat(r01, nfoldCV_tree(labels, features01, 0.9, 5));
        r10 = vcat(r10, nfoldCV_tree(labels, features10, 0.9, 5));
        r11 = vcat(r11, nfoldCV_tree(labels, features11, 0.9, 5));
    end
    #--------------------------------------------------


#   #--------------------------------------------------
#   # use random forest to predict empt
#   #--------------------------------------------------
#   r01  = Vector{Float64}();
#   r10  = Vector{Float64}();
#   r11 = Vector{Float64}();
#   for itr in 1:10
#       r01 = vcat(r01, nfoldCV_forest(labels, features01, 1, 10, 3, 5, 0.7));
#       r10 = vcat(r10, nfoldCV_forest(labels, features10, 1, 10, 3, 5, 0.7));
#       r11 = vcat(r11, nfoldCV_forest(labels, features11, 2, 10, 3, 5, 0.7));
#   end
#   #--------------------------------------------------

    println("\n\n(R2_10) = (", mean(r), ")");
    println("\n\n(R2_01, R2_10, R2_11) = (", mean(r01), ", ", mean(r10), ", ", mean(r11), ")");

    return code_vec, labels, features01, features10, model10;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function draw_decision_tree(tree, size_tree, max_level)
    #--------------------------------------------------
    g = DiGraph(size_tree);
    #--------------------------------------------------
    node_labels = Vector{String}();
    edge_labels = Dict{Tuple{Int64,Int64},String}();
    edge_styles = Dict{Tuple{Int64,Int64},String}();
    #--------------------------------------------------

    #--------------------------------------------------
    tree_nodes = Vector{Any}();
    #--------------------------------------------------
    push!(tree_nodes, tree);
    #--------------------------------------------------

    #--------------------------------------------------
    nid = 0;
    #--------------------------------------------------
    while (length(tree_nodes) != 0)
        #----------------------------------------------
        tree_node = splice!(tree_nodes,1); nid += 1;
        #----------------------------------------------
        if (typeof(tree_node) == DecisionTree.Node)
            #------------------------------------------
            node_label = latexstring("\\theta_{w} > " * string(round(tree_node.featval, 1)));
            #------------------------------------------
            push!(tree_nodes, tree_node.left);
            push!(tree_nodes, tree_node.right);
            #------------------------------------------
            add_edge!(g, nid, nid*2);
            add_edge!(g, nid, nid*2+1);
            #------------------------------------------
            edge_styles[(nid,nid*2)]   = "red";
            edge_styles[(nid,nid*2+1)] = "black!35!green";
            #------------------------------------------
        elseif (typeof(tree_node) == DecisionTree.Leaf)
            #------------------------------------------
            node_label = latexstring(string(@sprintf("%.2f", round(mean(tree_node.values)/1.0e6,2))) * "\\mathrm{M}");
            #------------------------------------------
        end
        #----------------------------------------------
        push!(node_labels, node_label);
        #----------------------------------------------
    end
    #--------------------------------------------------

    h = TikzGraphs.plot(g, node_labels, edge_labels=edge_labels, edge_styles=edge_styles);
    TikzPictures.save(SVG("results/decision_tree"), h);
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_brightkite(epsilon=1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true)
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
        if (!(isapprox(brightkite_dat[i,3],0.0) && isapprox(brightkite_dat[i,4],0.0)))
            if (!(brightkite_dat[i,1] in keys(id2no)))
                num_people += 1;
                no2id[num_people] = brightkite_dat[i,1];
                id2no[brightkite_dat[i,1]] = num_people;
            end

            id2lc[brightkite_dat[i,1]] = brightkite_dat[i,3:4];
        end
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
        coord = coord + rand(Normal(0.0,0.3),2);
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
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = 2.0;
    opt["delta_2"] = 0.2;

    if (epsilon > 0)
        @time theta, epsilon = SCP_FMM.model_fit(A, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        B = SCP_FMM.model_gen(theta, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        D = nothing;
    else
        error("option not supported.");
    end

    return A, B, theta, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_brightkite(A, theta, coords, option="degree", filename="output")
    h = Plots.plot(size=(570,350), title="Brightkite",
                             xlabel=L"\rm{Longitude}(^\circ)",
                             ylabel=L"\rm{Latitude}(^\circ)",
                             framestyle=:box,
                             grid="off");

    plot_core_periphery(h, A, theta, [flipdim(coord,1) for coord in coords], option;
                        plot_links=false,
                        distance="Haversine")

    Plots.savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_livejournal(epsilon=1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true)
    #--------------------------------
    # load airport data and location
    #--------------------------------
    user_dat = readdlm("data/livejournal/uid2crd");
    num_users = size(user_dat,1);
    #--------------------------------
    @assert length(user_dat[:,1]) == length(Set(user_dat[:,1]))
    #--------------------------------

    #--------------------------------
    no2id = Dict{Int64, Int64}();
    id2no = Dict{Int64, Int64}();
    id2lc = Dict{Int64, Array{Float64,1}}();
    #--------------------------------
    for i in 1:num_users
        no2id[i] = Int64(user_dat[i,1]);
        id2no[Int64(user_dat[i,1])] = i;
        #----------------------------
        id2lc[Int64(user_dat[i,1])] = user_dat[i,2:3];
    end
    #--------------------------------

    #--------------------------------
    I = Vector{Int64}();
    J = Vector{Int64}();
    V = Vector{Float64}();
    #--------------------------------
    # the adjacency matrix
    #--------------------------------
    edges_dat = convert(Array{Int64,2}, readdlm("data/livejournal/friendships"));
    num_edges = size(edges_dat,1);
    for i in 1:num_edges
        id1 = edges_dat[i,1];
        id2 = edges_dat[i,2];
        if (typeof(id1) == Int64 && typeof(id2) == Int64 && haskey(id2lc,id1) && haskey(id2lc,id2))
            push!(I, id2no[id1]);
            push!(J, id2no[id2]);
            push!(V, 1.0);
        end
    end
    #--------------------------------
    W = sparse(I,J,V, num_users,num_users,max);
    #--------------------------------
    W = W + W';
    #--------------------------------
    A = spones(sparse(W));
    #--------------------------------

    #--------------------------------
    # compute coordinates
    #--------------------------------
    coordinates = [];
    coords = zeros(2,num_users);
    for i in 1:num_users
        coord = id2lc[no2id[i]];
        coord = coord + rand(Normal(0.0,1.5),2);
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
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = 2.0;
    opt["delta_2"] = 0.2;

    dat = MAT.matread("results/livejournal1_distanceopt.mat");

    if (epsilon > 0)
        @time theta, epsilon = SCP_FMM.model_fit(A, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt, theta0=dat["C"]);
        B = SCP_FMM.model_gen(theta, coords, Haversine_CoM2, Haversine(6371e3), epsilon; opt=opt);
        D = nothing;
    else
        error("option not supported.");
    end

    return A, B, theta, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_livejournal(A, theta, coords, option="degree", filename="output")
    h = Plots.plot(size=(570,350), title="Livejournal",
                             xlabel=L"\rm{Longitude}(^\circ)",
                             ylabel=L"\rm{Latitude}(^\circ)",
                             framestyle=:box,
                             grid="off");

    plot_core_periphery(h, A, theta, [flipdim(coord,1) for coord in coords], option;
                        plot_links=false,
                        distance="Haversine")

    Plots.savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_mushroom(fname, epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true)
    #--------------------------------
    # load fungals data
    #--------------------------------
    dat = MAT.matread("data/fungal_networks/Conductance/" * fname);
#   data = MAT.matread("data/fungal_networks/Conductance/Pv_M_5xI_U_N_35d_1.mat");
    coords = dat["coordinates"]';
    A = spones(dat["A"]);
    #--------------------------------

    coordinates = [[coords[1,i], coords[2,i]] for i in 1:size(coords,2)];

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = 2.0;
    opt["delta_2"] = 0.2;

    #--------------------------------
    if (epsilon > 0)
        D = Euclidean_matrix(coordinates);
        # D = nothing;
        # theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
        # theta, epsilon = SCP_SGD.model_fit(A, D, epsilon; opt=opt);
        theta, epsilon = SCP_FMM.model_fit(A, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        # B = SCP.model_gen(theta, D, epsilon);
        B = SCP_FMM.model_gen(theta, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=Dict("ratio"=>0.0));
    elseif (epsilon < 0)
        D = rank_distance_matrix(Euclidean_matrix(coordinates));
        theta, epsilon = SCP.model_fit(A, D, -epsilon; opt=opt);
        B = SCP.model_gen(theta, D, epsilon);
    else
        D = ones(A)-eye(A);
        theta, epsilon = SCP.model_fit(A, D, 1; opt=opt);
        B = SCP.model_gen(theta, D, 1);
    end

    return A, B, theta, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_mushroom(A, theta, coords, option="degree", filename="output")
    h = Plots.plot(size=(570,450), title="Fungal Network (Pv_M_I_U_N_42d_1)",
                                   xlabel="x",
                                   ylabel="y",
                                   framestyle=:box,
                                   grid="off");

    plot_core_periphery(h, A, theta, coords, option;
                        plot_links=true,
                        distance="Euclidean")

    Plots.savefig(h, "results/" * filename * ".svg");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function analyze_mushroom()
    fnames = filter(x->contains(x,".mat"), readdir("results/fungal_networks/Conductance"));

    base4network  = Dict{String,Vector{Float64}}();
    theta4network = Dict{String,Vector{Float64}}();
    eps4network   = Dict{String,Float64}();

#   for fname in fnames
#       try
#           A,B,theta,D,coords,epsilon = test_mushroom(fname, 1.0; ratio=0.00, max_num_step=100, opt_epsilon=true);
#           MAT.matwrite("results/fungal_networks/Conductance/" * fname, Dict("A" => A, "B" => B, "C" => theta, "coords" => coords, "epsilon" => epsilon));
#
#           base4network[fname] = vec(sum(A,1));
#           theta4network[fname] = theta;
#           eps4network[fname] = epsilon;
#       catch y
#           println(y);
#       end
#   end

    xx = Vector{Vector{Float64}}();
    yy = Vector{String}();
    for fname in fnames
        #--------------------------------------------------
        dat = MAT.matread("results/fungal_networks/Conductance/" * fname);

        #--------------------------------------------------
        lg = Graph(dat["A"]);
        #--------------------------------------------------
        degree = vec(sum(dat["A"],1));
        btcetr = LightGraphs.betweenness_centrality(lg);
        clcetr = LightGraphs.closeness_centrality(lg);
        evcetr = LightGraphs.eigenvector_centrality(lg);
        pagerk = LightGraphs.pagerank(lg);
        #--------------------------------------------------

        base4network[fname]  = degree;
        theta4network[fname] = dat["C"];
        eps4network[fname]   = dat["epsilon"];

        xvec = Vector{Float64}();

        #-----------------------------------------
#       push!(xvec, length(theta4network[fname]));
        #-----------------------------------------
#       push!(xvec, mean(base4network[fname]));
#       push!(xvec, maximum(base4network[fname]));
#       push!(xvec, std(base4network[fname]));
        #-----------------------------------------
        push!(xvec, maximum(theta4network[fname]));
        push!(xvec, mean(theta4network[fname]));
#       push!(xvec, std(theta4network[fname]));
        #-----------------------------------------

        push!(xx, xvec);
        push!(yy, join(split(fname, "_")[1:end-2], "_"));
        #--------------------------------------------------
    end
    X = [xvec[i] for xvec in xx, i in 1:length(xx[1])];

    accuracy = cross_val_score(LogisticRegression(fit_intercept=true), X, yy; cv=5);

    println("accuracy: ", mean(accuracy));

    #------------------------------------------------------------
    h = Plots.plot(size=(350,260), title="Fungal Networks",
                                   xlabel="maximal vertex core score",
                                   ylabel="mean vertex core score",
                                   xlim = (4,24),
                                   xticks = 4:5:24,
                                   ylim = (0,15),
                                   framestyle=:box,
                                   legend=:none);
    #------------------------------------------------------------
    markershapes = [:circle, :utriangle, :dtriangle, :rect, :diamond];
    markercolors = [:red, :blue, :green];
    #------------------------------------------------------------
    max_cs = Vector()
    ave_cs = Vector()
    shp_mk = Vector()
    clr_mk = Vector()
    #------------------------------------------------------------
    for (i,label) in enumerate(unique(yy))
        #--------------------------------------------------------
        scatter!(h, [], [], markershape=markershapes[div(i-1,3)+1],
                            markercolor=markercolors[mod(i-1,3)+1],
                            markeralpha=0.6,
                            markerstrokewidth=0.1,
                            markerstrokecolor=markercolors[mod(i-1,3)+1],
                            markerstrokealpha=1.0,
                            label=label * " (" * string(sum(yy.==label)) * ")");
        #--------------------------------------------------------
        max_cs = vcat(max_cs, X[yy.==label, 1]);
        ave_cs = vcat(ave_cs, X[yy.==label, 2]);
        shp_mk = vcat(shp_mk, ones(Int64, sum(yy.==label)) * (div(i-1,3)+1));
        clr_mk = vcat(clr_mk, ones(Int64, sum(yy.==label)) * (mod(i-1,3)+1));
        #--------------------------------------------------------
    end
    #------------------------------------------------------------

    order = randperm(length(max_cs));

    #------------------------------------------------------------
    for id in order
        scatter!(h, [max_cs[id]], [ave_cs[id]], markershape=markershapes[shp_mk[id]],
                                                markercolor=markercolors[clr_mk[id]],
                                                markeralpha=0.6,
                                                markerstrokewidth=0.1,
                                                markerstrokecolor=markercolors[clr_mk[id]],
                                                markerstrokealpha=1.0,
                                                label="");
    end
    #------------------------------------------------------------

    Plots.savefig(h, "results/fungal_networks.svg");

    return base4network, theta4network, eps4network, X, yy, accuracy, h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_celegans(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true, delta_1=2.0, delta_2=0.2)
    #--------------------------------
    # load fungals data
    #--------------------------------
    data = MAT.matread("data/celegans/celegans277.mat");
#   data = MAT.matread("data/fungal_networks/Conductance/Pv_M_5xI_U_N_35d_1.mat");
    coords = data["celegans277positions"]';
    A = spones(convert(SparseMatrixCSC{Float64,Int64}, sparse(data["celegans277matrix"] + data["celegans277matrix"]')));
    #--------------------------------

    coordinates = [[coords[1,i], coords[2,i]] for i in 1:size(coords,2)];

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["max_num_step"] = max_num_step;
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = delta_1;
    opt["delta_2"] = delta_2;

    #--------------------------------
    if (epsilon > 0)
        D = Euclidean_matrix(coordinates);
#       theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
        t1 = time_ns()
        @time theta, epsilon, optim = SCP_FMM.model_fit(A, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        t2 = time_ns()

#       B = SCP.model_gen(theta, D, epsilon);
        B1 = SCP_FMM.model_gen(theta, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        B2 = SCP_FMM.model_gen(theta, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);
        B3 = SCP_FMM.model_gen(theta, coords, Euclidean_CoM2, Euclidean(), epsilon; opt=opt);

        B = [B1, B2, B3];
    elseif (epsilon < 0)
        D = rank_distance_matrix(Euclidean_matrix(coordinates));
        theta, epsilon = SCP.model_fit(A, D, -epsilon; opt=opt);
        B = SCP.model_gen(theta, D, epsilon);
    else
        D = ones(A)-eye(A);
        theta, epsilon = SCP.model_fit(A, D, 1; opt=opt);
        B = SCP.model_gen(theta, D, 1);
    end

    return A, B, theta, D, coordinates, epsilon, (t2-t1)/1.0e9/optim.f_calls;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function tradeoff_celegans(delta1_array, delta2_array)
    A,B,theta0,D,coordinates,epsilon,t = test_celegans(1.0; ratio=1.00, max_num_step=100, opt_epsilon=true, delta_1=0.0, delta_2=100.0);

    delta1_time = Vector{Float64}();
    delta1_rmse = Vector{Float64}();
    delta1_corr = Vector{Float64}();
    delta1_thta = Vector{Vector{Float64}}();
    delta1_epsl = Vector{Float64}();
    for delta1 in delta1_array
        A,B,theta,D,coordinates,epsilon,t = test_celegans(1.0; ratio=0.00, max_num_step=100, opt_epsilon=true, delta_1=delta1);
        push!(delta1_time, t);
        push!(delta1_rmse, (sum((theta-theta0).^2)/length(theta0))^0.5);
        push!(delta1_corr, cor(theta,theta0));
        push!(delta1_thta, theta);
        push!(delta1_epsl, epsilon);
    end

    delta2_time = Vector{Float64}();
    delta2_rmse = Vector{Float64}();
    delta2_corr = Vector{Float64}();
    delta2_thta = Vector{Vector{Float64}}();
    delta2_epsl = Vector{Float64}();
    for delta2 in delta2_array
        A,B,theta,D,coordinates,epsilon,t = test_celegans(1.0; ratio=0.00, max_num_step=100, opt_epsilon=true, delta_2=delta2);
        push!(delta2_time, t);
        push!(delta2_rmse, (sum((theta-theta0).^2)/length(theta0))^0.5);
        push!(delta2_corr, cor(theta,theta0));
        push!(delta2_thta, theta);
        push!(delta2_epsl, epsilon);
    end

    return delta1_time, delta1_rmse, delta1_corr, delta1_thta, delta1_epsl, delta2_time, delta2_rmse, delta2_corr, delta2_thta, delta2_epsl;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function accuracy_celegans(delta1_array, delta2_array)
    A,B,theta,D,coordinates,epsilon,t = test_celegans(1.0; ratio=1.00, max_num_step=100, opt_epsilon=true, delta_1=0.0, delta_2=100.0);
    omg_nev, omg_fmm, doe_nev, doe_fmm, epd_nev, epd_fmm, _, _ = check(A,theta,D,coordinates,Euclidean(),Euclidean_CoM2,epsilon,1.00,with_plot=false);

    delta_omg = zeros(length(delta1_array), length(delta2_array))
    delta_doe = zeros(length(delta1_array), length(delta2_array))
    delta_epd = zeros(length(delta1_array), length(delta2_array), length(epd_nev))
    delta_tog = zeros(length(delta1_array), length(delta2_array))
    delta_tgd = zeros(length(delta1_array), length(delta2_array))
    for (i,delta1) in enumerate(delta1_array)
        for (j,delta2) in enumerate(delta2_array)
            _, omg_fmm, _, doe_fmm, _, epd_fmm, t_omg, t_grd = check(A,theta,D,coordinates,Euclidean(),Euclidean_CoM2,epsilon,0.00,delta_1=delta1,delta_2=delta2,with_plot=false);
            delta_omg[i,j]   = omg_fmm
            delta_doe[i,j]   = doe_fmm
            delta_epd[i,j,:] = epd_fmm
            delta_tog[i,j]   = t_omg
            delta_tgd[i,j]   = t_grd

            println("(", i, ",", j, ") --- ", t_omg+t_grd)
        end
    end

    return delta_omg, delta_doe, delta_epd, delta_tog, delta_tgd, omg_nev, doe_nev, epd_nev
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_celegans(A, theta, coords, option="degree", filename="output")
    h = Plots.plot(size=(450,350), title="Celegans",
                             xlabel=L"x",
                             ylabel=L"y",
                             framestyle=:box);

    plot_core_periphery(h, A, theta, coords, option;
                        plot_links=true,
                        distance="Euclidean")

    Plots.savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function intro_celegans(A, theta, coords, option="community", filename="output")
    h = Plots.plot(size=(800,550), title="Celegans",
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
        vtx_sorted = sortperm(theta, rev=true);
        num_core = Int64(ceil(0.10*n));
        theta_max = theta[vtx_sorted[1]];
        theta_mid = theta[vtx_sorted[num_core]];
        theta_min = theta[vtx_sorted[end]];
        #--------------------------------------------------------
        Rmap = Colors.linspace(colorant"#ffe6e6", colorant"#ff0000", 100);
        Gmap = Colors.linspace(colorant"#008000", colorant"#e6ffe6", 100);
        Bmap = Colors.linspace(colorant"#0000ff", colorant"#e6e6ff", 100);
        #--------------------------------------------------------
        color = [(i in vtx_sorted[1:num_core] ? Rmap[Int64(ceil((theta[i]-theta_mid)/(theta_max-theta_mid) * 99)) + 1]
                                              : Bmap[Int64(ceil((theta[i]-theta_min)/(theta_mid-theta_min) * 99)) + 1]) for i in 1:n];
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
                    Plots.plot!(h, [coords[i][1], coords[j][1]],
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
    Plots.scatter!(h, [coord[1] for coord in coords], [coord[2] for coord in coords], ms=sqrt.(d)*1.8, c=color, alpha=1.00);
    #----------------------------------------------------------------

    Plots.savefig(h, "results/" * filename * ".svg");

    if (option == "community")
        order = [i for j in 1:10 for i in shuffle(1:n) if com[i] == j];
    elseif (option == "core_periphery")
        order = sortperm(theta, rev=true);
    end

    #----------------------------------------------------------------
    R = zeros(n,n);
    #----------------------------------------------------------------
    for i in 1:n
        for j in 1:n
            R[i,j] = A[order[i], order[j]];
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
    theta, epsilon = SCP.model_fit(A, D, 1; opt=Dict("thres"=>1.0e-6, "max_num_step"=>100));
    B = SCP.model_gen(theta, D, 1);
    #--------------------------------

    return A, B, theta, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function intro_Karate(A, theta, coords, option="community", filename="output")
    h = Plots.plot(size=(800,550), title="Karate",
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
        color = [(i in sortperm(theta, rev=true)[1:Int64(ceil(0.55*n))] ? colorant"#FF3333" : colorant"#33FFFF") for i in 1:n];
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
                    Plots.plot!(h, [coords[i][1], coords[j][1]],
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
    Plots.scatter!(h, [coord[1] for coord in coords], [coord[2] for coord in coords], ms=ones(n)*20, c=color, alpha=1.00);
    #----------------------------------------------------------------
    Plots.annotate!([(coords[i][1], coords[i][2], string(i), 15) for i in 1:n]);
    #----------------------------------------------------------------

    Plots.savefig(h, "results/" * filename * ".svg");

    if (option == "community")
        order = vcat(vec(com[2]), vec(com[1]));
    elseif (option == "core_periphery")
        order = sortperm(theta, rev=true);
    end

    #----------------------------------------------------------------
    R = zeros(n,n);
    #----------------------------------------------------------------
    for i in 1:n
        for j in 1:n
            R[i,j] = A[order[i], order[j]];
        end
    end
    #----------------------------------------------------------------

    return h, R;
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_facebook(epsilon=-1; ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true)
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
    opt["opt_epsilon"] = opt_epsilon;
    opt["delta_1"] = 2.0;
    opt["delta_2"] = 0.2;

    #--------------------------------
    if (epsilon > 0)
        D = Hamming_matrix(coordinates);
        theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
        # theta, epsilon = SCP_SGD.model_fit(A, D, epsilon; opt=opt);
        # theta, epsilon = SCP_FMM.model_fit(A, coords, Hamming_CoM2, Hamming(), epsilon; opt=opt);
        B = SCP.model_gen(theta, D, epsilon);
    elseif (epsilon < 0)
        D = rank_distance_matrix(Hamming_matrix(coordinates));
        theta, epsilon = SCP.model_fit(A, D, -epsilon; opt=opt);
        B = SCP.model_gen(theta, D, epsilon);
    else
        D = ones(A)-eye(A);
        theta, epsilon = SCP.model_fit(A, D, 1; opt=opt);
        B = SCP.model_gen(theta, D, 1);
    end

    return A, B, theta, D, coordinates, epsilon;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_facebook(A, theta, coords, option="degree", filename="output")
    h = Plots.plot(size=(600,600), title="Facebook",
                             xlabel=L"status",
                             ylabel=L"year");

    plot_core_periphery(h, A, theta, [[(coord[2] >=    0 ? coord[2] :    0) + (rand()-0.5)*0.3,
                                   (coord[7] >= 1999 ? coord[7] : 1999) + (rand()-0.5)*0.3] for coord in coords], option;
                        plot_links=true,
                        distance="Euclidean")

    Plots.savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_algo(sigma, num_vertices)
    h = Plots.plot(size=(600,600), title="",
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

    theta = ones(n) * (-4.0) + rand(n)*1.2;
    D = Euclidean_matrix(coords);
    A = SCP.model_gen(theta, D, 2);

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
                        Plots.plot!(h, [coords[i][1], coords[j][1]],
                                 [coords[i][2], coords[j][2]],
                                 legend=false,
                                 color="black",
                                 linewidth=2.0,
                                 linestyle=:dot,
                                 alpha=0.5);
                    else
                        Plots.plot!(h, [coords[i][1], coords[j][1]],
                                 [coords[i][2], coords[j][2]],
                                 legend=false,
                                 color="gray",
                                 linewidth=0.5,
                                 linestyle=:solid,
                                 alpha=0.3);
                    end
                end
                #------------------------------------------------
            end
        end
        #--------------------------------------------------------
    end
    #------------------------------------------------------------
    Plots.scatter!(h, [coord[1] for coord in coords], [coord[2] for coord in coords], ms=theta*5+23, c=colors, alpha=1.00);
    #----------------------------------------------------------------

    Plots.savefig(h, "results/algo_network.svg");

    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function check(A, theta, D, coordinates, metric, CoM2, epsilon, ratio; delta_1=2.0, delta_2=0.2, with_plot=true)
    coords = flipdim([coordinates[i][j] for i in 1:size(coordinates,1), j in 1:2]',1);
    bt = BallTree(coords, metric, leafsize=1);
    dist = Dict{Int64,Array{Float64,1}}(i => vec(D[:,i]) for i in 1:length(theta));

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

    omega_nev = SCP.omega(A, theta, D, epsilon);

    repeat = 1000

    t1 = time_ns()
    omega_fmm = SCP_FMM.omega!(theta, coords, CoM2, dist, epsilon, bt, A, sum_logD_inE, Dict("ratio" => ratio, "delta_1" => delta_1, "delta_2" => delta_2));
    for iter in 1:repeat
        omega_fmm = SCP_FMM.omega!(theta, coords, CoM2, dist, epsilon, bt, A, sum_logD_inE, Dict("ratio" => ratio, "delta_1" => delta_1, "delta_2" => delta_2));
    end
    t2 = time_ns()
    t_omg = (t2-t1)/1.0e9/repeat

    epd_nev = vec(sum(SCP.probability_matrix(theta, D, epsilon), 1));
    srd_nev = SCP.sum_rho_logD(theta,D,epsilon);

    t1 = time_ns()
    epd_fmm, srd_fmm, fmm_tree = SCP_FMM.epd_and_srd!(theta, coords, CoM2, dist, epsilon, bt, Dict("ratio" => ratio, "delta_1" => delta_1, "delta_2" => delta_2));
    for iter in 1:repeat
        epd_fmm, srd_fmm, fmm_tree = SCP_FMM.epd_and_srd!(theta, coords, CoM2, dist, epsilon, bt, Dict("ratio" => ratio, "delta_1" => delta_1, "delta_2" => delta_2));
    end
    t2 = time_ns()
    t_grd = (t2-t1)/1.0e9/repeat

    domega_depsilon_nev = (srd_nev-sum_logD_inE);
    domega_depsilon_fmm = (srd_fmm-sum_logD_inE);

    if (with_plot)
        order = sortperm(vec(sum(A,1)), rev=false);

        h = Plots.plot(size=(250,240), title="",
                                 xlabel="vertex indices",
                                 ylabel="expected degrees",
                                 xlim=(1,277),
                                 ylim=(-1.0, 80.0),
                                 grid="off",
                                 framestyle=:box,
                                 legend=:topleft);

        Plots.plot!(h, vec(sum(A,1))[order],          linestyle=:solid, linewidth=3.50, color="grey",   label="original degrees");
        Plots.plot!(h, epd_nev[order],                linestyle=:solid, linewidth=2.00, color="blue",   label="naive");
        Plots.plot!(h, epd_fmm[order],                linestyle=:solid, linewidth=1.00, color="orange", label="FMM");
        Plots.plot!(h, epd_fmm[order]-epd_nev[order], linestyle=:solid, linewidth=1.00, color="red",    label="FMM error");
        Plots.savefig(h, "results/expected_degrees.svg");

        return h, fmm_tree, omega_nev, omega_fmm, domega_depsilon_nev, domega_depsilon_fmm, epd_nev, epd_fmm;
    else
        return omega_nev, omega_fmm, domega_depsilon_nev, domega_depsilon_fmm, epd_nev, epd_fmm, t_omg, t_grd;
    end
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_cs_correlation(theta_nev, theta_fmm)
    h = Plots.plot(size=(250,240), title="",
                             xlabel="core scores (naive)",
                             ylabel="core scores (FMM)",
                             xlim=(-5.35,0.35),
                             ylim=(-5.35,0.35),
                             grid="off",
                             framestyle=:box,
                             legend=:topleft);

    Plots.plot!(h, -5.35:0.05:0.35, -5.35:0.05:0.35, color="red", label="ideal");
    Plots.scatter!(h, theta_nev, theta_fmm, label="experiment", color="blue", markerstrokewidth=0.3);

    Plots.savefig(h, "results/cs_correlation.svg");

    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function timeit(n, metric, CoM2, epsilon)
    #------------------------------------------------------------
    coords = rand(2,n);
    #------------------------------------------------------------
    bt = BallTree(coords, metric, leafsize=1);
    #------------------------------------------------------------

    TT = Dict(100 => -2.25, 1000 => -3.77, 10000 => -5.13, 100000 => -6.43, 1000000 => -7.69);

#   #------------------------------------------------------------
#   dmean = sqrt(3.0)/3.0;
#   #------------------------------------------------------------
#   function ff!(F, x, n)
#       F[1] = 0.0025 * (exp(2*x[1]+2.0) / (exp(2*x[1]+2.0) + dmean^epsilon)) +
#              0.0950 * (exp(2*x[1]+1.0) / (exp(2*x[1]+1.0) + dmean^epsilon)) +
#              0.9025 * (exp(2*x[1]+0.0) / (exp(2*x[1]+0.0) + dmean^epsilon)) - 10.0/n;
#   end
#   #------------------------------------------------------------
#   f!(F,x) = ff!(F,x,n);
#   #------------------------------------------------------------
#   C_p = nlsolve(f!,[0.0]).zero[1];
#   #------------------------------------------------------------
    theta = ones(n) * TT[n];
    #------------------------------------------------------------
    theta[1:Int64(ceil(0.05*n))] += 1.0;
    #------------------------------------------------------------

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
        @time [B_nev = SCP.model_gen(theta, D, epsilon)];
        @time [omega_nev = SCP.omega(B_nev, theta, D, epsilon)];
        @time [epd_nev = vec(sum(SCP.probability_matrix(theta, D, epsilon), 1)), srd = SCP.sum_rho_logD(theta,D,epsilon)];
        println(countnz(B_nev)/n);
    end

    @time [B_fmm = SCP_FMM.model_gen(theta, coords, CoM2, metric, epsilon; opt = Dict("ratio"=>0.0, "delta_1" => 2.0, "delta_2" => 0.2))];
    @time [omega_fmm = SCP_FMM.omega!(theta, coords, CoM2, Dict(), epsilon, bt, B_fmm, 0.0, Dict("ratio" => ratio, "delta_1" => 2.0, "delta_2" => 0.2))];
    @time [(epd_fmm, srd_fmm, fmm_tree) = SCP_FMM.epd_and_srd!(theta, coords, CoM2, Dict(), epsilon, bt, Dict("ratio" => ratio, "delta_1" => 2.0, "delta_2" => 0.2))];
    println(countnz(B_fmm)/n);
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_timings()
    #------------------------------------------------------------
    size = [1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6];
    #------------------------------------------------------------
    nev_gener = [0.000248, 0.016806,  2.462479,  246.000000,  24600.000000];
    nev_omega = [0.000672, 0.078040,  9.554892,  955.000000,  95500.000000];
    nev_deriv = [0.000836, 0.100245, 13.355793, 1336.000000, 133600.000000];
    #------------------------------------------------------------
    fmm_gener = [0.001963, 0.043340,  0.761756,    9.198479,    139.583723];
    fmm_omega = [0.000327, 0.004138,  0.058405,    0.784479,      7.757879];
    fmm_deriv = [0.000862, 0.014218,  0.224220,    2.639909,     26.322895];
    #------------------------------------------------------------

    h = Plots.plot(size=(330,270), title="Timings", xlabel="number of vertices",
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

    Plots.scatter!(h, size, fmm_omega, label="objective function", color="red", ms=6.5, markerstrokewidth=0.5);
    Plots.plot!(h, size, exp_omega, label=L"\mathcal{O}\left(|V| \cdot \log |V|\right)", color="red", linestyle=:dash, linewidth=2.0);
    Plots.scatter!(h, size, fmm_deriv, label="derivatives", color="blue", ms=6.5, markerstrokewidth=0.5);
    Plots.plot!(h, size, exp_deriv, label=L"\mathcal{O}\left(|V| \cdot \log |V|\right)", color="blue", linestyle=:dash, linewidth=2.0);
    Plots.scatter!(h, size, fmm_gener, label="generate network", color="green", ms=6.5, markerstrokewidth=0.5);
    Plots.plot!(h, size, exp_gener, label=L"\mathcal{O}\left(|V| \cdot (\log |V|)^2\right)", color="green", linestyle=:dash, linewidth=2.0);

    Plots.scatter!(h, size, fmm_omega, label="", color="red",   ms=6.5, markerstrokewidth=0.5);
    Plots.scatter!(h, size, fmm_deriv, label="", color="blue",  ms=6.5, markerstrokewidth=0.5);
    Plots.scatter!(h, size, fmm_gener, label="", color="green", ms=6.5, markerstrokewidth=0.5);

    Plots.savefig(h, "results/fmm_timings.svg");

    return h
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_ring(n, m, beta=0.0; epsilon=1.0, ratio=1.0, thres=1.0e-6, max_num_step=1000, opt_epsilon=true)
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
   opt["opt_epsilon"] = opt_epsilon;
   opt["delta_1"] = 2.0;
   opt["delta_2"] = 0.2;

   theta, epsilon = SCP.model_fit(A, D, epsilon; opt=opt);
   B = SCP.model_gen(theta, D, epsilon);
   #--------------------------------------------------------

   return A, B, theta, epsilon
end
#----------------------------------------------------------------
