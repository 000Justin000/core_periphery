#---------------------------------------------------------------------------------
module StochasticCP
    export model_fit, model_gen

    #-----------------------------------------------------------------------------
    # compute the probability matirx rho_{ij} denote probability for a link to
    # exist between node_i and node_j
    #-----------------------------------------------------------------------------
    function probability_matrix(C,D=ones(C.*C')-eye(C.*C'))
        @assert issymmetric(D);
    
        rho = exp.(C .+ C') ./ (exp.(C .+ C') .+ D);
        rho = rho - spdiagm(diag(rho));
    
        @assert issymmetric(rho);
    
        return rho;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # object function we are trying to maximize
    #-----------------------------------------------------------------------------
    # theta = \sum_{i<j} A_{ij} \log(rho_{ij}) + (1-A_{ij}) \log(1-rho_{ij})
    #-----------------------------------------------------------------------------
    function theta(A, C, D=ones(A)-eye(A))
        @assert issymmetric(A);
        @assert issymmetric(D);

        n = size(A,1);

        rho = probability_matrix(C,D)

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
    function model_fit(A, D=ones(A)-eye(A))
        @assert issymmetric(A);
        @assert issymmetric(D);
    
        A = spones(A);

        n = size(A,1);
        C = zeros(n);
        # C = rand(n);
        
        converged = false;
        while(!converged)
            C0 = copy(C);
    
            G = vec(sum(A-probability_matrix(C,D), 2));
            C = C + 0.001 * G;

            # println(C[1:5]);
    
            if (norm(C-C0)/norm(C) < 1.0e-6)
                converged = true;
            end
        end
    
        println(theta(A,C,D));
        return C;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # given the core score and distance matrix, compute the adjacency matrix
    #-----------------------------------------------------------------------------
    function model_gen(C, D=ones(C.*C')-eye(C.*C'))
        @assert issymmetric(D);

        n = size(C,1);
        rho = probability_matrix(C,D);

        A = spzeros(n,n);
        for i in 1:n
            for j in i+1:n
                A[i,j] = rand() < rho[i,j] ? 1 : 0;
            end
        end

        return A + A';
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
