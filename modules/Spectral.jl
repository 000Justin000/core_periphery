#---------------------------------------------------------------------------------
module Spectral
    export combinatorial_Laplacian, random_walk_Laplacian

    #-----------------------------------------------------------------------------
    # compute D - A, where D is the degree matrix, and A is the adjacency matrix
    #-----------------------------------------------------------------------------
    function combinatorial_Laplacian(A)
        if (!issymmetric(A))
            throw(ArgumentError("The input matrix must be symmetric."));
        else
            n = size(A,1);
        end
    
        D = spdiagm(vec(sum(A,1)));
    
        return D - A;
    end
    #-----------------------------------------------------------------------------

    #-----------------------------------------------------------------------------
    # compute D^{-1}A, where D is the degree matrix, and A is the adjacency matrix
    #-----------------------------------------------------------------------------
    function random_walk_Laplacian(A)
        if (!issymmetric(A))
            throw(ArgumentError("The input matrix must be symmetric."));
        else
            n = size(A,1);
        end
    
        d = vec(sum(A,1));
    
        P = copy(A);
    
        #--------------------------------------------------------------
        # if node i is isolated, then P_{ii} = 1
        #--------------------------------------------------------------
        for i in 1:n
            @assert d[i] >= 0;
    
            # P is a row stochastic matrix
            if (d[i] > 0)
                P[i,:] = P[i,:] ./ d[i];
            elseif (d[i] == 0)
                P[i,i] = 1;
            end
        end
    
        return P;
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
