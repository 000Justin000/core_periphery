#---------------------------------------------------------------------------------
module StochasticCP_FMM
    using StatsBase;
    using Distances;
    using NearestNeighbors;
#   using Plots; pyplot();

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
            if (idx*2+1 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
                fill_cm!(cmp, idx*2,   bt, ms, roid, srid, CoM2);
                fill_cm!(cmp, idx*2+1, bt, ms, roid, srid, CoM2);
                cmp[idx] = Particle(CoM2(cmp[idx*2].CoM,cmp[idx*2+1].CoM, cmp[idx*2].m,cmp[idx*2+1].m), cmp[idx*2].m + cmp[idx*2+1].m, 0.0);
            elseif (idx*2 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
                fill_cm!(cmp, idx*2,   bt, ms, roid, srid, CoM2);
                cmp[idx] = Particle(cmp[idx*2].CoM, cmp[idx*2].m, cmp[idx*2].pot);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    function subtree_size(idx, n)
        p = floor(log(n)/log(2)) - floor(log(idx)/log(2));

        if (2^p * idx + 2^p - 1 <= n)
            size = 2^(p+1) - 1;
        elseif (2^p * idx <= n)
            size = (2^p - 1) + (n - 2^p * idx + 1);
        else
            size = (2^p - 1);
        end

        return size;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the potential between two nodes
    #-----------------------------------------------------------------------------
    function fill_p2!(cmp, idx_1, idx_2, bt, eplison)
        n_node = bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs;
        if ((idx_1 <= n_node) && (idx_2 <= n_node))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center)
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            if (distance >= max(eplison,2)*(sp1r + sp2r))
                if ((idx_1 > bt.tree_data.n_internal_nodes) && (idx_2 > bt.tree_data.n_internal_nodes))
                    cmp[idx_1].pot += cmp[idx_2].m / (cmp[idx_1].m * cmp[idx_2].m + distance^eplison);
                    cmp[idx_2].pot += cmp[idx_1].m / (cmp[idx_1].m * cmp[idx_2].m + distance^eplison);
                else
                    cmp[idx_1].pot += cmp[idx_2].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^eplison);
                    cmp[idx_2].pot += cmp[idx_1].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^eplison);
#                   cmp[idx_1].pot += cmp[idx_2].m / distance^eplison;
#                   cmp[idx_2].pot += cmp[idx_1].m / distance^eplison;
                end
                cmp[end].pot += 1
            elseif (sp1r <= sp2r)
                fill_p2!(cmp, idx_1, idx_2*2,   bt, eplison)
                fill_p2!(cmp, idx_1, idx_2*2+1, bt, eplison)
            else
                fill_p2!(cmp, idx_1*2,   idx_2, bt, eplison)
                fill_p2!(cmp, idx_1*2+1, idx_2, bt, eplison)
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively compute the potential at each level of the tree
    #-----------------------------------------------------------------------------
    function fill_p!(cmp, idx, bt, eplison)
        if (idx <= bt.tree_data.n_internal_nodes)
            fill_p2!(cmp, idx*2, idx*2+1, bt, eplison)
            fill_p!(cmp,  idx*2,          bt, eplison)
            fill_p!(cmp,  idx*2+1,        bt, eplison)
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
    function expected_degree(C::Array{Float64,1}, coords, CoM2, dist, eplison, bt, ratio)
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

            cid2all = exp.(C[cid] .+ C) ./ (exp.(C[cid] .+ C) .+ dist[cid].^eplison);
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
        fill_p!(fmm_tree, 1, bt, eplison)
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

        return epd, fmm_tree;
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # given the adjacency matrix and coordinates, compute the core scores
    #-----------------------------------------------------------------------------
    function model_fit(A::SparseMatrixCSC{Float64,Int64},
                       coords::Array{Float64,2},
                       CoM2,
                       metric = Euclidean(),
                       eplison = 1;
                       opt = Dict("thres"=>1.0e-6, "step_size"=>0.01, "max_num_step"=>10000, "ratio"=>1.0))
        @assert issymmetric(A);
        A = spones(A);
        n = size(A,1);
        d = vec(sum(A,2));
        order = sortperm(d, rev=true);
        C = d / maximum(d) * 1.0e-6;

        converged = false;
        num_step = 0;
        acc_step = 0;
        acc_C    = zeros(n);

        dist = Dict{Int64,Array{Float64,1}}()
        bt = BallTree(coords, metric, leafsize=1);

        delta_C = 1.0;
        step_size = opt["step_size"];

        while(!converged && num_step < opt["max_num_step"])
            num_step += 1;

            C0 = copy(C);

            # compute the expected degree with fmm;
            epd, fmm_tree = expected_degree(C, coords, CoM2, dist, eplison, bt, opt["ratio"]);

            # compute the gradient
            G = d - epd;

#           h = plot(C[order]);
#           display(h);

            # update the core score
            C = C + G * step_size + 0.0 * (rand(n)*2-1) * step_size;

            if (norm(C-C0)/norm(C) < opt["thres"])
                converged = true;
            else
                if (norm(C-C0)/norm(C) > 0.99 * delta_C)
                    step_size *= 0.99;
                end
                delta_C = norm(C-C0)/norm(C);

                println(num_step, ": ", delta_C);
            end

            if (num_step > 0.9 * opt["max_num_step"] - 1)
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
        for j in 1:n
            for i in j+1:n
                A[i,j] = rand() < exp(C[i]+C[j])/(exp(C[i]+C[j]) + D[i,j]) ? 1 : 0;
            end
        end

        return A + A';
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
