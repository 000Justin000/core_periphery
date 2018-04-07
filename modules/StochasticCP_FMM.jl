#---------------------------------------------------------------------------------
module StochasticCP_FMM
    using StatsBase;
    using Distances;
    using NearestNeighbors;

    export model_fit, model_gen


    #-----------------------------------------------------------------------------
    # data structure providing information of a given node
    #-----------------------------------------------------------------------------
    mutable struct Particle
        CoM::Vector{Float64}
        m::Float64
        pot::Float64
    end
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # recursively compute the center of mass of each node
    #-----------------------------------------------------------------------------
    function fill_cm!(cmp, idx, bt, ms, roid, srid, CoM2)
        if (idx > bt.tree_data.n_internal_nodes)
            cmp[idx] = Particle(convert(Vector{Float64}, bt.hyper_spheres[idx].center), ms[roid[srid[idx]]], 0.0)
        else
            fill_cm!(cmp, idx*2,   bt, ms, roid, srid, CoM2)
            fill_cm!(cmp, idx*2+1, bt, ms, roid, srid, CoM2)
    
            cmp[idx] = Particle(CoM2(cmp[idx*2].CoM,cmp[idx*2+1].CoM, cmp[idx*2].m,cmp[idx*2+1].m), cmp[idx*2].m + cmp[idx*2+1].m, 0.0)
        end
    end
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # compute the potential between two nodes
    #-----------------------------------------------------------------------------
    function fill_p2!(cmp, idx_1, idx_2, bt, od)
        if ((idx_1 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs) &&
            (idx_2 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center)
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            if (distance >= 2*(sp1r + sp2r))
                cmp[idx_1].pot += cmp[idx_2].m / distance^od
                cmp[idx_2].pot += cmp[idx_1].m / distance^od
                cmp[end].pot += 1
            elseif (sp1r <= sp2r)
                fill_p2!(cmp, idx_1, idx_2*2,   bt, od)
                fill_p2!(cmp, idx_1, idx_2*2+1, bt, od)
            else
                fill_p2!(cmp, idx_1*2,   idx_2, bt, od)
                fill_p2!(cmp, idx_1*2+1, idx_2, bt, od)
            end
        end
    end
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # recursively compute the potential at each level of the tree
    #-----------------------------------------------------------------------------
    function fill_p!(cmp, idx, bt, od)
        if (idx <= bt.tree_data.n_internal_nodes)
            fill_p2!(cmp, idx*2, idx*2+1, bt, od)
            fill_p!(cmp,  idx*2,          bt, od)
            fill_p!(cmp,  idx*2+1,        bt, od)
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively accumulate the potential to the lowest level
    #-----------------------------------------------------------------------------
    function accumulate_p!(cmp, idx, bt)
        if (2 <= idx <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
            cmp[idx].pot += cmp[div(idx,2)].pot
        end

        if (idx <= bt.tree_data.n_internal_nodes)
            accumulate_p!(cmp, idx*2,   bt)
            accumulate_p!(cmp, idx*2+1, bt)
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the expected degree of each node with fast multipole method
    #-----------------------------------------------------------------------------
    function expected_degree(C::Array{Float64,1}, coords, CoM2, dist, od, bt, ratio)
        n = length(C)

        #-------------------------------------------------------------------------
        # expected degree
        #-------------------------------------------------------------------------
        epd = zeros(n)
        #-------------------------------------------------------------------------
        core_id = sortperm(C, rev=true)[1:Int64(ceil(ratio * n))]
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # compute the expected degree exactly for core nodes
        #-------------------------------------------------------------------------
        epdc = Dict{Int64, Float64}()
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # now compute the c->c and c->p
        #-------------------------------------------------------------------------
        for cid in core_id
            if (!haskey(dist, cid))
                dist2cid = zeros(n);
                for i in 1:n
                    dist2cid[i] = evaluate(bt.metric, coords[:,cid], coords[:,i]);
                end

                dist[cid] = dist2cid;
            end

            cid2all = exp.(C[cid] .+ C) ./ (exp.(C[cid] .+ C) .+ dist[cid].^od);
            cid2all[cid] = 0;
            epd += cid2all;

            epdc[cid] = sum(cid2all);
        end
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # now compute the p->c and p->p
        #-------------------------------------------------------------------------
        td = bt.tree_data;
        ni = td.n_internal_nodes;
        nl = td.n_leafs;
        roid = bt.indices;             # (o)riginal id of the (r)eordered data points
        orid = sortperm(roid);         # (r)eordered id of the (o)riginal data points
        #-------------------------------------------------------------------------
        # (r)eordered data point id of the hyper(s)pheres
        srid = Dict((idx >= td.cross_node) ? (idx => td.offset_cross + idx) : (idx => td.offset + idx) for idx in (ni+1:ni+nl));
        # hyper(s)phere id of the (r)eordered data points
        rsid = Dict((idx >= td.cross_node) ? (td.offset_cross + idx => idx) : (td.offset + idx => idx) for idx in (ni+1:ni+nl));
        #-------------------------------------------------------------------------
        # data structure that stores the node's CoM, mass, and potential 
        # corresponding to nodes in BallTree data structure
        #-------------------------------------------------------------------------
        fmm_tree = Array{Particle,1}(ni+nl+1);
        fmm_tree[end] = Particle([0.0,0.0],0.0,0.0);
        #-------------------------------------------------------------------------
        #-------------------------------------------------------------------------
        ms = exp.(C)
        #-------------------------------------------------------------------------
        for cid in core_id
            ms[cid] = 0
        end
        #-------------------------------------------------------------------------
        fill_cm!(fmm_tree, 1, bt, ms, roid, srid, CoM2)
        fill_p!(fmm_tree, 1, bt, od)
        accumulate_p!(fmm_tree, 1, bt)
        #-------------------------------------------------------------------------
        epd += [fmm_tree[rsid[orid[idx]]].pot for idx in 1:nl] .* exp.(C)
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # replace with the (stored) exact expected degree for core nodes
        #-------------------------------------------------------------------------
        for cid in core_id
            epd[cid] = epdc[cid];
        end
        #-------------------------------------------------------------------------

        return epd;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # theta = \sum_{i<j} A_{ij} \log(rho_{ij}) + (1-A_{ij}) \log(1-rho_{ij})
    #-----------------------------------------------------------------------------
    function theta(A, C, D)
        @assert issymmetric(A);
        @assert issymmetric(D);

        n = size(A,1);

        rho = probability_matrix(C,D,1:n)

        theta = 0;
        for i in 1:n
            for j in i+1:n
                theta += A[i,j]*log(rho[i,j]) + (1-A[i,j])*log(1-rho[i,j]);
            end
        end

        return theta;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # given the adjacency matrix and coordinates, compute the core scores
    #-----------------------------------------------------------------------------
    function model_fit(A::SparseMatrixCSC{Float64,Int64},
                       coords::Array{Float64,2},
                       CoM2,
                       metric = Euclidean(),
                       od = 1;
                       opt = Dict("thres"=>1.0e-6, "step_size"=>0.01, "max_num_step"=>10000, "ratio"=>1.0))
        @assert issymmetric(A);
        A = spones(A);
        n = size(A,1);
        D = vec(sum(A,2));
        C = D / maximum(D) * 1.0e-6;

        converged = false;
        num_step = 0;
        acc_step = 0;
        acc_C    = zeros(n);

        dist = Dict{Int64,Array{Float64,1}}()
        bt = BallTree(coords, Haversine(6317e3), leafsize=1);

        while(!converged && num_step < opt["max_num_step"])
            num_step += 1;

            C0 = copy(C);

            # compute the gradient with FMM
            G = D - expected_degree(C, coords, CoM2, dist, od, bt, opt["ratio"])

            # update the core score
            C = C + 0.5*G*opt["step_size"] + (rand(n)*2-1)*opt["step_size"];

            if (norm(C-C0)/norm(C) < opt["thres"])
                converged = true;
            else
                println(num_step, ": ", norm(C-C0)/norm(C));
            end

            if (num_step > 0.7 * opt["max_num_step"])
                acc_step += 1;
                acc_C    += C;
            end
        end

        CC = acc_step > 0 ? acc_C/acc_step : C;
    
        return CC;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # given the core score and distance matrix, compute the adjacency matrix
    #-----------------------------------------------------------------------------
    function model_gen(C, D=ones(C.*C')-eye(C.*C'))
        @assert issymmetric(D);

        n = size(C,1);

        A = spzeros(n,n);
        for i in 1:n
            for j in i+1:n
                A[i,j] = rand() < exp(C[i]+C[j])/(exp(C[i]+C[j]) + D[i,j]) ? 1 : 0;
            end
        end

        return A + A';
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
