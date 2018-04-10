using StatsBase;
using MAT;
using NetworkGen;
using StochasticCP_FMM;
using Motif;
using Colors;
using NearestNeighbors;
using Distances;
using Plots; pyplot();
using LaTeXStrings;

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
            C   = model_fit(A);
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
    C = model_fit(A);

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
    C  = model_fit(W0);
    W1 = model_gen(C);

    plot(C);

    return W0, W1, data;
end
#----------------------------------------------------------------

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
function Haversine_matrix(coords)
    n = size(coords,1);
    D = zeros(n,n);

    for j in 1:n
        for i in j+1:n
            D[i,j] = haversine(flipdim(coords[i],1), flipdim(coords[j],1), 6371e3)
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
        println("rank_distance_matrix: ", j);
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

    D = vec(sum(A,1));
    
    if (option == "degree")
        color = [(i in sortperm(D, rev=true)[1:Int64(ceil(0.1*n))] ? colorant"orange" : colorant"blue") for i in 1:n];
    elseif (option == "core_score")
        color = [(i in sortperm(C, rev=true)[1:Int64(ceil(0.1*n))] ? colorant"orange" : colorant"blue") for i in 1:n];
    else
        error("option not supported.");
    end

    if (option == "degree")
        rk = sortperm(sortperm(D, rev=true))
        ms = ((rk-1)/n - 1).^20 * 6 + 0.3;
    elseif (option == "core_score")
        rk = sortperm(sortperm(C, rev=true))
        ms = ((rk-1)/n - 1).^20 * 6 + 0.3;
    else
        error("option not supported.");
    end

    #------------------------------------------------------------
    if (plot_links == true)
        #------------------------------------------------------------
        if (distance == "Euclidean")
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
                                 alpha=0.15);
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
    scatter!(h, [coord[1] for coord in coords], [coord[2] for coord in coords], ms=ms, c=color, label="");
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_underground(distance_option="no_distance"; ratio=1.0, thres=1.0e-6, step_size=0.01, max_num_step=10000)
    data = MAT.matread("data/london_underground/london_underground_clean.mat");

    W = [Int(sum(list .!= 0)) for list in data["Labelled_Network"]];
    A = spones(sparse(W));

    D = Haversine_matrix(data["Tube_Locations"]);

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["step_size"] = step_size;
    opt["max_num_step"] = max_num_step;

    if (distance_option == "no_distance")
        C = model_fit(A; opt=opt);
        B = model_gen(C);
    elseif (distance_option == "distance")
        C = model_fit(A, D; opt=opt);
        B = model_gen(C, D);
    elseif (distance_option == "rank_distance")
        C = model_fit(A, rank_distance_matrix(D); opt=opt);
        B = model_gen(C, rank_distance_matrix(D));
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
        ms = D*2;
    elseif (option == "core_score")
        ms=(2.^C/maximum(2.^C))*10;
    else
        error("option not supported.");
    end

    coords = data["Tube_Locations"];
    h = plot(size=(600,400), title="London Underground", 
                             xlabel=L"\rm{Latitude} (^\circ)", 
                             ylabel=L"\rm{Longitude}(^\circ)");
    for i in 1:n
        for j in i+1:n
            if (A[i,j] != 0)
                h = plot!([coords[i][1], coords[j][1]], [coords[i][2], coords[j][2]], leg=false, color="black", linewidth=1);
            end
        end
    end
    h = scatter!([coord[1] for coord in coords], [coord[2] for coord in coords], ms=ms, c=color);
    savefig(h, "results/" * filename * ".pdf");

    return h;
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
    # png(h, "results/air_rank_distance_hist")

    return h
end
#----------------------------------------------------------------


#----------------------------------------------------------------
function test_openflight(dist_opt=-1; ratio=1.0, thres=1.0e-6, step_size=0.01, max_num_step=1000)
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

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["step_size"] = step_size;
    opt["max_num_step"] = max_num_step;

    if (dist_opt >= 0)
        D = Haversine_matrix(coordinates).^dist_opt;
        # C = model_fit(A, D; opt=opt);
        C = model_fit(A, coords, Haversine_CoM2, Haversine(6371e3), 2; opt=opt);
        B = model_gen(C, D);
    elseif (distance_option == -1)
        D = rank_distance_matrix(Haversine_matrix(coordinates));
        C = model_fit(A, D; opt=opt);
        B = model_gen(C, D);
    else
        error("distance_option not supported");
    end

    return A, B, C, Haversine_matrix(coordinates), coordinates
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_openflight(A, C, coords, option="degree", filename="output")
    h = plot(size=(1200,650), title="Openflight", 
                              xlabel=L"\rm{Latitude} (^\circ)", 
                              ylabel=L"\rm{Longitude}(^\circ)");

    plot_core_periphery(h, A, C, [flipdim(coord,1) for coord in coords], "degree";
                        plot_links=true,
                        distance="Haversine")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_mushroom(dist_opt=-1; ratio=1.0, thres=1.0e-6, step_size=0.01, max_num_step=1000)
    #--------------------------------
    # load fungals data
    #--------------------------------
    data = MAT.matread("data/fungal_networks/Conductance/Ag_M_I+4R_U_N_42d_1.mat");
    coords = data["coordinates"]';
    A = spones(data["A"]);
    #--------------------------------

    coordinates = [[coords[1,i], coords[2,i]] for i in 1:size(coords,2)];

    opt = Dict()
    opt["ratio"] = ratio;
    opt["thres"] = thres;
    opt["step_size"] = step_size;
    opt["max_num_step"] = max_num_step;

    #--------------------------------
    if (dist_opt >= 0)
        D = Euclidean_matrix(coordinates).^dist_opt;
        # C = model_fit(A, D; opt=opt);
        C = model_fit(A, coords, Euclidean_CoM2, Euclidean(), dist_opt; opt=opt);
        B = model_gen(C, D);
    elseif (distance_option == -1)
        D = rank_distance_matrix(Euclidean_matrix(coordinates));
        C = model_fit(A, D; opt=opt);
        B = model_gen(C, D);
    else
        error("distance_option not supported");
    end

    return A, B, C, Euclidean_matrix(coordinates), coordinates
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function plot_mushroom(A, C, coords, option="degree", filename="output")
    h = plot(size=(1200,650), title="Mushroom",
                              xlabel=L"x", 
                              ylabel=L"y");

    plot_core_periphery(h, A, C, [flipdim(coord,1) for coord in coords], "degree";
                        plot_links=true,
                        distance="Euclidean")

    savefig(h, "results/" * filename * ".pdf");
    return h;
end
#----------------------------------------------------------------


function check(C, D, coordinates, dist_opt, ratio)
    coords = [coordinates[i][j] for i in 1:size(coordinates,1), j in 1:2]';
    bt = BallTree(coords, Euclidean(), leafsize=1);
    dist = Dict{Int64,Array{Float64,1}}(i => vec(D[:,i]) for i in 1:length(C));
    epd_real = vec(sum(StochasticCP.probability_matrix(C, D.^dist_opt), 1));
    epd      = StochasticCP_FMM.expected_degree(C, coords, Euclidean_CoM2, dist, dist_opt, bt, ratio);

    order = sortperm(C, rev=false);
    plot(epd_real[order])
    plot!(epd[order])
    plot!(epd[order] - epd_real[order])
end
