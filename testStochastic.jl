using MAT;
using NetworkGen;
using StochasticCP;
using Motif;
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
function distance_matrix(coord)
    n = size(coord,1);
    D = spzeons(n,n);

    for i in 1:n
        for j in i+1:n
            lat1 = coord[i][1]/180 * pi;
            lat2 = coord[j][1]/180 * pi;
            lon1 = coord[i][2]/180 * pi;
            lon2 = coord[j][2]/180 * pi;

            dlat = lat2-lat1;
            dlon = lon2-lon1;

            haversine = sin(dlat/2)^2 + cos(lat1)*cos(lat2)*sin(dlon/2)^2;
            D[i,j] = 6371e3 * (2 * atan2(sqrt(haversine), sqrt(1-haversine)));
        end
    end

    D = D + D';

    @assert issymmetric(D);

    return D;
end
#----------------------------------------------------------------

#----------------------------------------------------------------
function test_underground()
    data = MAT.matread("data/london_underground/London_Underground.mat");

    A = [Int(sum(list .!= 0)) for list in data["Labelled_Network"]];
    A = spones(A);

    D = distance_matrix(data["Tube_Locations"])

    C = StochasticCP.model_fit(A);
    B = StochasticCP.model_gen(C);

    plot(C);

    return A, B, data;
end
#----------------------------------------------------------------
