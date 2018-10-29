#---------------------------------------------------------------------------------
module SCP_SGD
    using StatsBase

    export model_fit, model_gen

    #-----------------------------------------------------------------------------
    # compute the probability matirx rho_{:,slt}, only for selected columns in slt
    #-----------------------------------------------------------------------------
    function probability_matrix(C,Dslt,slt,eplison)
        rho = exp.(C .+ C[slt]') ./ (exp.(C .+ C[slt]') .+ Dslt.^eplison);

        for i in 1:size(slt,1)
            rho[slt[i],i] = 0;
        end

        return rho;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # theta = \sum_{i<j} A_{ij} \log(rho_{ij}) + (1-A_{ij}) \log(1-rho_{ij})
    #-----------------------------------------------------------------------------
    function theta(A, C, D, eplison)
        @assert issymmetric(A);
        @assert issymmetric(D);

        n = size(A,1);

        rho = probability_matrix(C,D,1:n,eplison)

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
    # given the adjacency matrix and distance matrix, compute the core scores
    #-----------------------------------------------------------------------------
    function model_fit(A, D, eplison; opt=Dict("thres"=>1.0e-6,
                                                "step_size"=>0.01,
                                                "max_num_step"=>10000,
                                                "ratio"=>1.0))
        @assert issymmetric(A);
        @assert issymmetric(D);

        A = spones(A);
        n = size(A,1);
        d = vec(sum(A,2));
        order = sortperm(d,rev=true);
        C = d / maximum(d) * 1.0e-6;

        converged = false;
        num_step = 0;
        acc_step = 0;
        acc_C    = zeros(n);

        while(!converged && num_step < opt["max_num_step"])
            num_step += 1;

            C0 = copy(C);

            # sample ratio*n nodes to be active
            slt = sample(1:n, Int64(ceil(opt["ratio"]*n)), replace=false, ordered=true);
            # compute the gradient with respect to the sampled node
            G = vec(sum(A[:,slt]-probability_matrix(C,D[:,slt],slt,eplison), 2)) * (n/Int64(ceil(opt["ratio"]*n)));

            # update the core score
            C = C + G * opt["step_size"] + 0.5 * (rand(n)*2-1) * opt["step_size"];

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

        println(theta(A,CC,D,eplison));
        return CC;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # given the core score and distance matrix, compute the adjacency matrix
    #-----------------------------------------------------------------------------
    function model_gen(C, D, eplison)
        @assert issymmetric(D);

        n = size(C,1);

        A = spzeros(n,n);
        for j in 1:n
            for i in j+1:n
                A[i,j] = rand() < exp(C[i]+C[j])/(exp(C[i]+C[j]) + D[i,j]^eplison) ? 1 : 0;
            end
        end

        return A + A';
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
