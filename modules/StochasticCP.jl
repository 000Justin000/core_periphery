#---------------------------------------------------------------------------------
module StochasticCP
using Optim
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
    # omega = \sum_{i<j} A_{ij} \log(rho_{ij}) + (1-A_{ij}) \log(1-rho_{ij})
    #-----------------------------------------------------------------------------
    function omega(A, C, D, epsilon)
        @assert issymmetric(A);
        @assert issymmetric(D);

        n = size(A,1);
        rho = probability_matrix(C,D,epsilon)

        omega = 0;
        for i in 1:n
            for j in i+1:n
                omega += A[i,j]*log(rho[i,j]) + (1-A[i,j])*log(1-rho[i,j]);
            end
        end

        return omega;
    end
    #-----------------------------------------------------------------------------


    #-----------------------------------------------------------------------------
    # gradient of objective function
    #-----------------------------------------------------------------------------
    function negative_gradient_omega!(A, C, D, epsilon, sum_logD_inE, storage)
        @assert issymmetric(A);
        @assert issymmetric(D);

        G = vec(sum(A-probability_matrix(C,D,epsilon), 2));
        srd = sum_rho_logD(C,D,epsilon);

        storage[1:end-1] = -G;
        storage[end] = -(srd - sum_logD_inE);
    end
    #-----------------------------------------------------------------------------


    #---------------------------------------------------------------------------------------------
    # given the adjacency matrix and distance matrix, compute the core scores
    #---------------------------------------------------------------------------------------------
    # if epsilon is integer, then fix epsilon, otherwise optimize epsilon as well as core_score
    #---------------------------------------------------------------------------------------------
    function model_fit(A, D, epsilon; opt=Dict("thres"=>1.0e-6,
                                                "max_num_step"=>10000))
        @assert issymmetric(A);
        @assert issymmetric(D);

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
                sum_logD_inE += log(D[i,j])
            end
            #---------------------------------------------------------------------
        end
        #-----------------------------------------------------------------------------

        f(x)           = -omega(A,x[1:end-1],D,x[end]);
        g!(storage, x) =  negative_gradient_omega!(A,x[1:end-1],D,x[end],sum_logD_inE,storage)

        #-----------------------------------------------------------------------------
        println("starting optimization:")
        #-----------------------------------------------------------------------------
        precond = speye(length(C)+1); precond[end,end] = length(C);
        optim = optimize(f, g!, vcat(C,[epsilon]), LBFGS(P = precond), Optim.Options(g_tol = opt["thres"],
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

        println(omega(A,C,D,epsilon));
        @assert epsilon > 0;
        return C, epsilon;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute the gradient with respect to the order of distance
    #-----------------------------------------------------------------------------
    function sum_rho_logD(C, D, epsilon)
        @assert issymmetric(D);
        @assert sum(abs.(diag(D))) == 0;

        n = length(C);
        rho = probability_matrix(C,D,epsilon);

        #-----------------------------------------------------------------------------
        sum_rho_logD = 0.0;
        #-----------------------------------------------------------------------------
        for i in 1:n
            for j in 1:n
                if (i != j)
                    sum_rho_logD += rho[i,j] * log(D[i,j]);
                end
            end
#           if (i in 1:1)
#               println(sum_rho_logD);
#               println(D[i,1:5]);
#               println(C[1:5]);
#           end
        end
        #-----------------------------------------------------------------------------

        return sum_rho_logD/2.0;
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
