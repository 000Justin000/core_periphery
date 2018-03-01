#---------------------------------------------------------------------------------
module NetworkGen
    export cc_cp_pp, di_cc_cp_pp

    #-----------------------------------------------------------------------------
    # generate a random undirected network
    #-----------------------------------------------------------------------------
    # n          - number of nodes
    # cratio     - ratio of core nodes
    # pcc        - probability of core-core linkage
    # pcp        - probability of core-periphery linkage
    # ppp        - probability of periphery-periphery linkage
    # rand_order - if true, then core nodes and periphery nodes are mixed
    #-----------------------------------------------------------------------------
    function cc_cp_pp(n, cratio, pcc, pcp, ppp, rand_order=false)
        if (rand_order)
            vertices = randperm(n);
        else
            vertices = 1:n;
        end
    
        core = vertices[1:Int(floor(n*cratio))];
        peri = vertices[Int(floor(n*cratio))+1:end];
    
        A = spzeros(n,n);
    
        for i in 1:n
            for j in i+1:n
                if ((i in core) && (j in core))
                    A[i,j] = rand() < pcc ? 1 : 0;
                elseif ((i in peri) && (j in peri))
                    A[i,j] = rand() < ppp ? 1 : 0;
                else
                    A[i,j] = rand() < pcp ? 1 : 0;
                end
            end
        end
    
        return A + A';
    end
    #-----------------------------------------------------------------------------
    
    #-----------------------------------------------------------------------------
    # generate a random undirected network
    #-----------------------------------------------------------------------------
    # n          - number of nodes
    # cratio     - ratio of core nodes
    # pcc        - probability of core-core linkage
    # pcp        - probability of core-periphery linkage
    # ppp        - probability of periphery-periphery linkage
    # rand_order - if true, then core nodes and periphery nodes are mixed
    #-----------------------------------------------------------------------------
    function di_cc_cp_pp(n, cratio, pcc, pcp, ppp, rand_order=false)
        if (rand_order)
            vertices = randperm(n);
        else
            vertices = 1:n;
        end
    
        core = vertices[1:Int(floor(n*cratio))];
        peri = vertices[Int(floor(n*cratio))+1:end];
    
        A = spzeros(n,n);
    
        for i in 1:n
            for j in 1:n
                if (i != j)
                    if ((i in core) && (j in core))
                        A[i,j] = rand() < pcc ? 1 : 0;
                    elseif ((i in peri) && (j in peri))
                        A[i,j] = rand() < ppp ? 1 : 0;
                    else
                        A[i,j] = rand() < pcp ? 1 : 0;
                    end
                end
            end
        end
    
        return A;
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
