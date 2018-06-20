#---------------------------------------------------------------------------------
module StochasticCP_FMM
    using StatsBase;
    using Distances;
    using NearestNeighbors;
    using Optim;
#   using Plots; pyplot();

    export model_fit, model_gen


    #-----------------------------------------------------------------------------
    # data structure providing information of a given node
    #-----------------------------------------------------------------------------
    mutable struct Particle
        CoM::Vector{Float64}
        m::Float64
        pot_1::Float64
        pot_2::Float64
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # recursively compute the center of mass of each node
    #-----------------------------------------------------------------------------
    function fill_cm!(cmp, idx, bt, ms, roid, srid, CoM2)
        if (idx > bt.tree_data.n_internal_nodes)
            cmp[idx] = Particle(convert(Vector{Float64}, bt.hyper_spheres[idx].center), ms[roid[srid[idx]]], 0.0, 0.0);
        else
            if (idx*2+1 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
                fill_cm!(cmp, idx*2,   bt, ms, roid, srid, CoM2);
                fill_cm!(cmp, idx*2+1, bt, ms, roid, srid, CoM2);
                cmp[idx] = Particle(CoM2(cmp[idx*2].CoM,cmp[idx*2+1].CoM, cmp[idx*2].m,cmp[idx*2+1].m), cmp[idx*2].m + cmp[idx*2+1].m, 0.0, 0.0);
            elseif (idx*2 <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
                fill_cm!(cmp, idx*2,   bt, ms, roid, srid, CoM2);
                cmp[idx] = Particle(cmp[idx*2].CoM, cmp[idx*2].m, cmp[idx*2].pot_1, cmp[idx*2].pot_2);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the potential between two nodes
    #-----------------------------------------------------------------------------
    function acc_p2!(cmp, idx_1, idx_2, bt, epsilon)
        n_node = bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs;
        if ((idx_1 <= n_node) && (idx_2 <= n_node))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center);
            #-----------------------------------------------------------------
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            #-----------------------------------------------------------------
            if ((idx_1 > bt.tree_data.n_internal_nodes) && (idx_2 > bt.tree_data.n_internal_nodes))
                cmp[end].pot_1 += log(1 + (cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon));
                cmp[end].m += 1;
            elseif (distance >= max(epsilon*2, 2)*(sp1r + sp2r) && ((cmp[idx_1].m * cmp[idx_2].m)/(distance^epsilon) < 0.2))
            # elseif ((sp1r + sp2r) < 1.0e-12)
                cmp[end].pot_1 += +(1/1) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^1
                                  -(1/2) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^2
                                  +(1/3) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^3
                                  -(1/4) * ((cmp[idx_1].m * cmp[idx_2].m) / (distance^epsilon))^4;
                cmp[end].m += 1;
            elseif (sp1r <= sp2r)
                acc_p2!(cmp, idx_1, idx_2*2,   bt, epsilon);
                acc_p2!(cmp, idx_1, idx_2*2+1, bt, epsilon);
            else
                acc_p2!(cmp, idx_1*2,   idx_2, bt, epsilon);
                acc_p2!(cmp, idx_1*2+1, idx_2, bt, epsilon);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively compute the potential at each level of the tree
    #-----------------------------------------------------------------------------
    function acc_p!(cmp, idx, bt, epsilon)
        if (idx <= bt.tree_data.n_internal_nodes)
            acc_p2!(cmp, idx*2, idx*2+1, bt, epsilon)
            acc_p!(cmp,  idx*2,          bt, epsilon)
            acc_p!(cmp,  idx*2+1,        bt, epsilon)
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the expected degree of each node with fast multipole method
    #-----------------------------------------------------------------------------
    function omega!(C::Array{Float64,1}, coords, CoM2, dist, epsilon, bt, ratio, A, sum_logD_inE)
        n = length(C);

        #-------------------------------------------------------------------------
        omega = 0.0;
        #-------------------------------------------------------------------------
        I,J,V = findnz(A);
        #-----------------------------------------------------------------------------
        for (i,j) in zip(I,J)
            #---------------------------------------------------------------------
            if (i < j)
                omega += C[i] + C[j];
            end
            #---------------------------------------------------------------------
        end
        #-----------------------------------------------------------------------------
        omega -= epsilon * sum_logD_inE;
        #-----------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        core_id = sortperm(C, rev=true)[1:Int64(ceil(ratio * n))]; # JJ: debug
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

            #---------------------------------------------------------------------
            cid2all = log.( (exp.(C[cid] .+ C) .+ dist[cid].^epsilon) ./ (dist[cid].^epsilon) );
            cid2all[cid] = 0;     omega -= 0.5 * sum(cid2all);     # c->c and c->p
            cid2all[core_id] = 0; omega -= 0.5 * sum(cid2all);              # c->p
            #---------------------------------------------------------------------
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
        ms = exp.(C); ms[core_id] = 0;
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        fmm_tree = Array{Particle,1}(ni+nl+1);
        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0);
        #-------------------------------------------------------------------------
        fill_cm!(fmm_tree, 1, bt, ms, roid, srid, CoM2);
        acc_p!(fmm_tree, 1, bt, epsilon);
        #-------------------------------------------------------------------------
        omega -= fmm_tree[end].pot_1;
        #-------------------------------------------------------------------------

#        #-------------------------------------------------------------------------
#        fmm_tree = Array{Particle,1}(ni+nl+1);
#        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0);
#        #-------------------------------------------------------------------------
#        fill_cm!(fmm_tree, 1, bt, ms.^2, roid, srid, CoM2);
#        acc_p!(fmm_tree, 1, bt, epsilon*2);
#        #-------------------------------------------------------------------------
#        omega -= (-1/2)*fmm_tree[end].pot_1;
#        #-------------------------------------------------------------------------
#        println(omega);
#
#        #-------------------------------------------------------------------------
#        fmm_tree = Array{Particle,1}(ni+nl+1);
#        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0);
#        #-------------------------------------------------------------------------
#        fill_cm!(fmm_tree, 1, bt, ms.^3, roid, srid, CoM2);
#        acc_p!(fmm_tree, 1, bt, epsilon*3);
#        #-------------------------------------------------------------------------
#        omega -= (1/3)*fmm_tree[end].pot_1;
#        #-------------------------------------------------------------------------
#        println(omega);

        return omega;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    function subtree_range(idx, n)
        @assert mod(n,2) == 1;
        @assert idx <= n;

        p = Int64(floor(log(n)/log(2)) - floor(log(idx)/log(2)));

        if (2^p * idx + 2^p - 1 <= n) # the tree that root at idx is full binary tree with height p
            range = collect(2^p * idx : 2^p * idx + 2^p - 1);
        elseif (2^p * idx <= n)       # the tree that root at idx is (not full) binary tree with height p
            range = vcat(collect(2^(p-1) * idx + div(n - 2^p * idx + 1, 2) : 2^(p-1) * idx + 2^(p-1) - 1), collect(2^p * idx : n));
        else                          # the tree that root at idx is full binary tree with height p-1
            range = collect(2^(p-1) * idx : 2^(p-1) * idx + 2^(p-1) - 1);
        end

        return range;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    function subtree_size(idx, n)
        @assert mod(n,2) == 1;
        @assert idx <= n;

        p = Int64(floor(log(n)/log(2)) - floor(log(idx)/log(2)));

        if (2^p * idx + 2^p - 1 <= n) # the tree that root at idx is full binary tree with height p
            size = 2^p;
        elseif (2^p * idx <= n)       # the tree that root at idx is (not full) binary tree with height p
            size = 2^(p-1) + div(n - 2^p * idx + 1, 2);
        else                          # the tree that root at idx is full binary tree with height p-1
            size = 2^(p-1);
        end

        return size;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the potential between two nodes
    #-----------------------------------------------------------------------------
    function fill_p2!(cmp, idx_1, idx_2, bt, epsilon)
        n_node = bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs;
        if ((idx_1 <= n_node) && (idx_2 <= n_node))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center);
            #-----------------------------------------------------------------
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            #-----------------------------------------------------------------
            if ((idx_1 > bt.tree_data.n_internal_nodes) && (idx_2 > bt.tree_data.n_internal_nodes))
                cmp[idx_1].pot_1 += cmp[idx_2].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon);
                cmp[idx_2].pot_1 += cmp[idx_1].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon);
                cmp[idx_1].pot_2 += cmp[idx_2].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon) * log(distance);
                cmp[idx_2].pot_2 += cmp[idx_1].m / (cmp[idx_1].m * cmp[idx_2].m + distance^epsilon) * log(distance);
                cmp[end].m += 1;
            elseif (distance >= max(epsilon*2, 2)*(sp1r + sp2r))
            # elseif ((sp1r + sp2r) < 1.0e-12)
                # cmp[idx_1].pot_1 += cmp[idx_2].m / distance^epsilon;
                # cmp[idx_2].pot_1 += cmp[idx_1].m / distance^epsilon;
                cmp[idx_1].pot_1 += cmp[idx_2].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon);
                cmp[idx_2].pot_1 += cmp[idx_1].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon);
                cmp[idx_1].pot_2 += cmp[idx_2].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon) * log(distance);
                cmp[idx_2].pot_2 += cmp[idx_1].m / ((cmp[idx_1].m * cmp[idx_2].m) / (subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon) * log(distance);
                cmp[end].m += 1;
            elseif (sp1r <= sp2r)
                fill_p2!(cmp, idx_1, idx_2*2,   bt, epsilon);
                fill_p2!(cmp, idx_1, idx_2*2+1, bt, epsilon);
            else
                fill_p2!(cmp, idx_1*2,   idx_2, bt, epsilon);
                fill_p2!(cmp, idx_1*2+1, idx_2, bt, epsilon);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively compute the potential at each level of the tree
    #-----------------------------------------------------------------------------
    function fill_p!(cmp, idx, bt, epsilon)
        if (idx <= bt.tree_data.n_internal_nodes)
            fill_p2!(cmp, idx*2, idx*2+1, bt, epsilon)
            fill_p!(cmp,  idx*2,          bt, epsilon)
            fill_p!(cmp,  idx*2+1,        bt, epsilon)
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively accumulate the potential to the lowest level
    #-----------------------------------------------------------------------------
    function accumulate_p!(cmp, idx, bt)
        if (2 <= idx <= bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs)
            cmp[idx].pot_1 += cmp[div(idx,2)].pot_1
            cmp[idx].pot_2 += cmp[div(idx,2)].pot_2
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
    function epd_and_srd!(C::Array{Float64,1}, coords, CoM2, dist, epsilon, bt, ratio)
        n = length(C);

        #-------------------------------------------------------------------------
        # "expected degree" and "sum_rho_logD"
        #-------------------------------------------------------------------------
        epd = zeros(n);
        srd = 0.0;
        #-------------------------------------------------------------------------
        core_id = sortperm(C, rev=true)[1:Int64(ceil(ratio * n))]; # JJ: debug
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # compute the expected degree and sum_rho_logD exactly for core nodes
        #-------------------------------------------------------------------------
        epdc = Dict{Int64, Float64}();
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

            #---------------------------------------------------------------------
            cid2all_1 = exp.(C[cid] .+ C) ./ (exp.(C[cid] .+ C) .+ dist[cid].^epsilon);
            cid2all_1[cid] = 0; epd += cid2all_1;
            epdc[cid] = sum(cid2all_1);                      # store c->c and c->p
            #---------------------------------------------------------------------
            cid2all_2 = exp.(C[cid] .+ C) ./ (exp.(C[cid] .+ C) .+ dist[cid].^epsilon) .* log.(dist[cid]);
            cid2all_2[cid] = 0;     srd += sum(cid2all_2);         # c->c and c->p
            cid2all_2[core_id] = 0; srd += sum(cid2all_2);         # p->c
            #---------------------------------------------------------------------
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
        ms = exp.(C); ms[core_id] = 0;
        #-------------------------------------------------------------------------
        fmm_tree = Array{Particle,1}(ni+nl+1);
        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0);
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        fill_cm!(fmm_tree, 1, bt, ms, roid, srid, CoM2);
        fill_p!(fmm_tree, 1, bt, epsilon);
        accumulate_p!(fmm_tree, 1, bt);
        #-------------------------------------------------------------------------
        epd += [fmm_tree[rsid[orid[idx]]].pot_1 for idx in 1:nl] .* ms;
        #-------------------------------------------------------------------------
        srd += sum([fmm_tree[rsid[orid[idx]]].pot_2 for idx in 1:nl] .* ms);
        #-------------------------------------------------------------------------
        srd /= 2.0;
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        # replace with the (stored) exact expected degree for core nodes
        #-------------------------------------------------------------------------
        for cid in core_id
            epd[cid] = epdc[cid];
        end
        #-------------------------------------------------------------------------

        return epd, srd, fmm_tree;
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # gradient of objective function
    #-----------------------------------------------------------------------------
    function negative_gradient_omega!(C, coords, CoM2, dist, epsilon, bt, ratio, d, sum_logD_inE, storage)
        epd, srd, fmm_tree = epd_and_srd!(C, coords, CoM2, dist, epsilon, bt, ratio);

        G = d - epd;

        storage[1:end-1] = -G;
        storage[end] = -(srd - sum_logD_inE);
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # given the adjacency matrix and coordinates, compute the core scores
    #-----------------------------------------------------------------------------
    function model_fit(A::SparseMatrixCSC{Float64,Int64},
                       coords::Array{Float64,2},
                       CoM2,
                       metric = Euclidean(),
                       epsilon = 1;
                       opt = Dict("thres"=>1.0e-6, "max_num_step"=>10000, "ratio"=>1.0))
        @assert issymmetric(A);
        A = spones(A);
        n = size(A,1);
        d = vec(sum(A,2));
        order = sortperm(d, rev=true);
        C = d / maximum(d) * 1.0e-6;

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
                sum_logD_inE += log(evaluate(metric, coords[:,i], coords[:,j]))
            end
            #---------------------------------------------------------------------
        end
        #-----------------------------------------------------------------------------

        dist = Dict{Int64,Array{Float64,1}}()
        bt = BallTree(coords, metric, leafsize=1);

        f!(x)          = -omega!(x[1:end-1], coords, CoM2, dist, x[end], bt, opt["ratio"], A, sum_logD_inE);
        g!(storage, x) =  negative_gradient_omega!(x[1:end-1], coords, CoM2, dist, x[end], bt, opt["ratio"], d, sum_logD_inE, storage)

        #-----------------------------------------------------------------------------
        println("starting optimization:")
        #-----------------------------------------------------------------------------
        # lo = -ones(length(C)+1) * Inf; lo[end] = 0;
        # hi =  ones(length(C)+1) * Inf;
        #
        # od = OnceDifferentiable(f!,g!,vcat(C,[epsilon]));
        # optim = optimize(od, vcat(C,[epsilon]), lo, hi, Fminbox{LBFGS}(), show_trace = true,
        #                                                                   show_every = 1,
        #                                                                   allow_f_increases = true,
        #                                                                   iterations = opt["max_num_step"]);
        #-----------------------------------------------------------------------------
        precond = speye(length(C)+1); precond[end,end] = length(C);
        optim = optimize(f!, g!, vcat(C,[epsilon]), LBFGS(P = precond), Optim.Options(g_tol = 1e-6,
                                                                                      iterations = opt["max_num_step"],
                                                                                      show_trace = true,
                                                                                      show_every = 1,
                                                                                      allow_f_increases = false));
        #-----------------------------------------------------------------------------
        println(optim);
        #-----------------------------------------------------------------------------


        C = optim.minimizer[1:end-1];
        epsilon = optim.minimizer[end];

        println(epsilon);
        println(omega!(C, coords, CoM2, dist, epsilon, bt, opt["ratio"], A, sum_logD_inE));

        @assert epsilon > 0;
        return C, epsilon;
    end
    #-----------------------------------------------------------------------------



    #-----------------------------------------------------------------------------
    # generate edges between points within two hyper_spheres
    #-----------------------------------------------------------------------------
    function gen_e2!(cmp, idx_1, idx_2, bt, epsilon, roid, srid, A)
        n_node = bt.tree_data.n_internal_nodes + bt.tree_data.n_leafs;
        if ((idx_1 <= n_node) && (idx_2 <= n_node))
            #-----------------------------------------------------------------
            distance = evaluate(bt.metric, bt.hyper_spheres[idx_1].center, bt.hyper_spheres[idx_2].center);
            #-----------------------------------------------------------------
            sp1r = bt.hyper_spheres[idx_1].r
            sp2r = bt.hyper_spheres[idx_2].r
            #-----------------------------------------------------------------
            if ((idx_1 > bt.tree_data.n_internal_nodes) && (idx_2 > bt.tree_data.n_internal_nodes))
                @assert roid[srid[idx_1]] != roid[srid[idx_2]];
                if (A[roid[srid[idx_1]], roid[srid[idx_2]]] != 1)
                    A[roid[srid[idx_1]], roid[srid[idx_2]]] = rand() < (cmp[idx_1].m * cmp[idx_2].m)/((cmp[idx_1].m * cmp[idx_2].m) + distance^epsilon) ? 1 : 0;
                end
            elseif (distance >= max(epsilon*2, 2)*(sp1r + sp2r))
            # elseif ((sp1r + sp2r) < 1.0e-12)
                nef = (cmp[idx_1].m * cmp[idx_2].m) / ((cmp[idx_1].m * cmp[idx_2].m)/(subtree_size(idx_1,n_node)*subtree_size(idx_2,n_node)) + distance^epsilon);
                nei = Int64(floor(nef) + rand() < (nef - floor(nef)) ? 1 : 0);
                # generate ne edges between this two group of nodes
                grp_1 = subtree_range(idx_1,n_node);
                grp_2 = subtree_range(idx_2,n_node);

                grp_1_mass = [cmp[it].m for it in grp_1];
                grp_2_mass = [cmp[it].m for it in grp_2];

                grp_1_mass0 = mean(grp_1_mass);
                grp_2_mass0 = mean(grp_2_mass);

                grp_1_prob = (grp_1_mass .* grp_2_mass0) ./ (grp_1_mass .* grp_2_mass0 + distance^epsilon); grp_1_prob /= sum(grp_1_prob);
                grp_2_prob = (grp_2_mass .* grp_1_mass0) ./ (grp_2_mass .* grp_1_mass0 + distance^epsilon); grp_2_prob /= sum(grp_2_prob);

                grp_1_bin = vcat([0], cumsum(grp_1_prob)[1:end-1]);
                grp_2_bin = vcat([0], cumsum(grp_2_prob)[1:end-1]);

                offset = rand() * 1.0/nei;
                # offset = 0;
                for i in 0:nei-1
                    target = i/nei + offset;
                    id_1 = searchsortedlast(grp_1_bin, target);
                    id_2 = searchsortedlast(grp_2_bin*grp_1_prob[id_1], target-grp_1_bin[id_1]);

                    @assert roid[srid[grp_1[id_1]]] != roid[srid[grp_2[id_2]]];
                    @assert A[roid[srid[grp_1[id_1]]], roid[srid[grp_2[id_2]]]] != 1;
                    A[roid[srid[grp_1[id_1]]], roid[srid[grp_2[id_2]]]] = 1;
                end

                cmp[end].m += 1;
            elseif (sp1r <= sp2r)
                gen_e2!(cmp, idx_1, idx_2*2,   bt, epsilon, roid, srid, A);
                gen_e2!(cmp, idx_1, idx_2*2+1, bt, epsilon, roid, srid, A);
            else
                gen_e2!(cmp, idx_1*2,   idx_2, bt, epsilon, roid, srid, A);
                gen_e2!(cmp, idx_1*2+1, idx_2, bt, epsilon, roid, srid, A);
            end
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # recursively generate edges at each level of the tree
    #-----------------------------------------------------------------------------
    function gen_e!(cmp, idx, bt, epsilon, roid, srid, A)
        if (idx <= bt.tree_data.n_internal_nodes)
            gen_e2!(cmp, idx*2, idx*2+1, bt, epsilon, roid, srid, A);
            gen_e!(cmp,  idx*2,          bt, epsilon, roid, srid, A);
            gen_e!(cmp,  idx*2+1,        bt, epsilon, roid, srid, A);
        end
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # given the core score and distance matrix, compute the adjacency matrix
    #-----------------------------------------------------------------------------
    function model_gen(C::Array{Float64,1},
                       coords::Array{Float64,2},
                       CoM2,
                       metric = Euclidean(),
                       epsilon = 1;
                       opt = Dict("ratio"=>1.0))

        n = length(C);
        A = spzeros(n,n)

        dist = Dict{Int64,Array{Float64,1}}();
        bt = BallTree(coords, metric, leafsize=1);

        #-------------------------------------------------------------------------
        core_id  = sortperm(C, rev=true)[1:Int64(ceil(opt["ratio"] * n))];
        core_set = Set(core_id);
        #-------------------------------------------------------------------------
        # now compute the c->c and c->p
        #-------------------------------------------------------------------------
        println(length(core_id));
        #-------------------------------------------------------------------------
        for cid in core_id
            if (!haskey(dist, cid))
                dist2cid = zeros(n);
                for i in 1:n
                    dist2cid[i] = evaluate(bt.metric, coords[:,cid], coords[:,i]);
                end

                dist[cid] = dist2cid;
            end

            for i in 1:n
                if (!(i in core_set && i <= cid))
                    @assert A[cid,i] == 0;
                    @assert A[i,cid] == 0;
                    A[cid,i] = rand() < exp(C[cid]+C[i])/(exp(C[cid]+C[i]) + dist[cid][i]^epsilon) ? 1 : 0;
                end
            end
        end
        #-------------------------------------------------------------------------
        println(sum(A));
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
        ms = exp.(C); ms[core_id] = 0;
        #-------------------------------------------------------------------------
        fmm_tree = Array{Particle,1}(ni+nl+1);
        fmm_tree[end] = Particle([0.0,0.0], 0.0, 0.0, 0.0);
        #-------------------------------------------------------------------------

        #-------------------------------------------------------------------------
        fill_cm!(fmm_tree, 1, bt, ms, roid, srid, CoM2);
        gen_e!(fmm_tree, 1, bt, epsilon, roid, srid, A);
        #-------------------------------------------------------------------------
        println(sum(A));
        #-------------------------------------------------------------------------

        return A + A';
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
