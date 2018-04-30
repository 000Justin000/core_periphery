#---------------------------------------------------------------------------------
module StochasticCP
#   using Plots; pyplot();

    export model_fit, model_gen

    #-----------------------------------------------------------------------------
    # compute the probability matirx rho_{ij} denote probability for a link to
    # exist between node_i and node_j
    #-----------------------------------------------------------------------------
    function probability_matrix(C, D, epsilon)
        @assert issymmetric(D);

        rho = exp.(C .+ C') ./ (exp.(C .+ C') .+ D.^epsilon);
        rho = rho - diagm(diag(rho));

        @assert issymmetric(rho);

        return rho;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # object function we are trying to maximize
    #-----------------------------------------------------------------------------
    # theta = \sum_{i<j} A_{ij} \log(rho_{ij}) + (1-A_{ij}) \log(1-rho_{ij})
    #-----------------------------------------------------------------------------
    function theta(A, C, D, epsilon)
        @assert issymmetric(A);
        @assert issymmetric(D);

        n = size(A,1);
        rho = probability_matrix(C,D,epsilon)

        theta = 0;
        for i in 1:n
            for j in i+1:n
                theta += A[i,j]*log(rho[i,j]) + (1-A[i,j])*log(1-rho[i,j]);
            end
        end

        return theta;
    end
    #-----------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------
    # given the adjacency matrix and distance matrix, compute the core scores
    #---------------------------------------------------------------------------------------------
    # if epsilon is integer, then fix epsilon, otherwise optimize epsilon as well as core_score
    #---------------------------------------------------------------------------------------------
    function model_fit(A, D, epsilon; opt=Dict("thres"=>1.0e-6,
                                                "step_size"=>0.01,
                                                "max_num_step"=>10000))
        @assert issymmetric(A);
        @assert issymmetric(D);

        A = spones(A);
        n = size(A,1);
        d = vec(sum(A,2));
        order = sortperm(d, rev=true);
        C = d / maximum(d) * 1.0e-6;

        # C = rand(n);
        # C = 0.5 * ones(n) * log((sum(A)/n^2)/(1 - sum(A)/n^2) * median(D));

        converged = false;
        num_step = 0;

        delta_C = 1.0;
        step_size = opt["step_size"];

        while(!converged && num_step < opt["max_num_step"])
            num_step += 1;
            C0 = copy(C);

            # compute the gradient
            G = vec(sum(A-probability_matrix(C,D,epsilon), 2));

            C = C + step_size * G;

            if (typeof(epsilon) <: AbstractFloat)
                eps_grd  = 1.0e-2 * step_size * epsilon_gradient(A,C,D,epsilon);
                epsilon += abs(eps_grd) < step_size ? eps_grd : sign(eps_grd) * step_size;
            end

#           h = plot(C[order]);
#           display(h);

            if (norm(C-C0)/norm(C) < opt["thres"])
                converged = true;
            else
                if (norm(C-C0)/norm(C) > 0.99 * delta_C)
                    step_size *= 0.99;
                end
                delta_C = norm(C-C0)/norm(C);

                println(num_step, ": ", epsilon, "  ", step_size, "  ", eps_grd, "  ", delta_C);
            end
        end

        println(theta(A,C,D,epsilon));
        return C;
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # compute the gradient with respect to the order of distance
    #-----------------------------------------------------------------------------
    function epsilon_gradient(A, C, D, epsilon)
        @assert issymmetric(A);
        @assert issymmetric(D);
        @assert sum(abs.(diag(D))) == 0;

        rho = probability_matrix(C,D,epsilon);
        dep = -rho.^2 .* exp.(-(C .+ C')) .* log.(D + eye(D)) .* D.^epsilon;

        n = size(A,1);

        @assert issymmetric(dep);


        #-----------------------------------------------------------------------------
        epsilon_gradient = 0;
        #-----------------------------------------------------------------------------
        for i in 1:n
            for j in i+1:n
                epsilon_gradient += (A[i,j]/rho[i,j] - (1-A[i,j])/(1-rho[i,j])) * dep[i,j];
            end
        end
        #-----------------------------------------------------------------------------

        return epsilon_gradient;
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # given the core score and distance matrix, compute the adjacency matrix
    #-----------------------------------------------------------------------------
    function model_gen(C, D, epsilon)
        @assert issymmetric(D);

        n = size(C,1);

        A = spzeros(n,n);
        for j in 1:n
            for i in j+1:n
                A[i,j] = rand() < exp(C[i]+C[j])/(exp(C[i]+C[j]) + D[i,j]^epsilon) ? 1 : 0;
            end
        end

        return A + A';
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
