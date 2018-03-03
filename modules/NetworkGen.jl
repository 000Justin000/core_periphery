#---------------------------------------------------------------------------------
module NetworkGen
    export n_r_cc_cp_pp, di_n_r_cc_cp_pp

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
    function n_r_cc_cp_pp(n, cratio, pcc, pcp, ppp, rand_order=false)
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
    # generate a random directed network
    #-----------------------------------------------------------------------------
    # n          - number of nodes
    # cratio     - ratio of core nodes
    # pcc        - probability of core-core linkage
    # pcp        - probability of core-periphery linkage
    # ppp        - probability of periphery-periphery linkage
    # rand_order - if true, then core nodes and periphery nodes are mixed
    #-----------------------------------------------------------------------------
    function di_n_r_cc_cp_pp(n, cratio, pcc, pcp, ppp, rand_order=false)
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

    #-----------------------------------------------------------------------------
    # generate a lattice network
    #-----------------------------------------------------------------------------
    function periodic_lattice(n, m)
        A = spzeros(n*m,n*m);

        pp(itr,len) = Int((itr-1) .- floor.((itr-1)./len) .* len + 1);

        for i in 1:n
            for j in 1:m
                A[(pp(i,n)-1)*n + (pp(j,m)-1) + 1, (pp(i-1,n)-1)*n + (pp(j,  m)-1) + 1] = 1;
                A[(pp(i,n)-1)*n + (pp(j,m)-1) + 1, (pp(i+1,n)-1)*n + (pp(j,  m)-1) + 1] = 1;
                A[(pp(i,n)-1)*n + (pp(j,m)-1) + 1, (pp(i,  n)-1)*n + (pp(j-1,m)-1) + 1] = 1;
                A[(pp(i,n)-1)*n + (pp(j,m)-1) + 1, (pp(i,  n)-1)*n + (pp(j+1,m)-1) + 1] = 1;
            end
        end

        @assert issymmetric(A);
        return A;
    end
    #-----------------------------------------------------------------------------
end
#---------------------------------------------------------------------------------
