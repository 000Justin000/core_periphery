#---------------------------------------------------------------------------------
module Motif
    export Me0, Me1, Me2, M05, M13

    #-------------------------
    # && edge
    #-------------------------
    function Me0(A)
        A = spones(A);
        B = min.(A, A');
        
        @assert issymmetric(B);
        return B;
    end
    #-------------------------

    #-------------------------
    # || edge
    #-------------------------
    function Me1(A)
        A = spones(A);
        C = max.(A, A');
        
        @assert issymmetric(C);
        return C;
    end
    #-------------------------

    #-------------------------
    # XOR edge
    #-------------------------
    function Me2(A)
        A = spones(A);
        B = min.(A, A');
        U = A - B;
        
        @assert issymmetric(U);
        return U;
    end
    #-------------------------

    #-------------------------
    # feed forward loop
    #-------------------------
    function M05(A)
        A = spones(A);
        B = min.(A, A');
        U = A - B;

        T1 = (U  * U ) .* U;
        T2 = (U' * U ) .* U;
        T3 = (U  * U') .* U;
        W  = T1 + T2 + T3;
        W  = W  + W';
        
        @assert issymmetric(W);
        return W;
    end
    #-------------------------

    #-------------------------
    # two hop
    #-------------------------
    function M13(A)
        A = spones(A);
        B = min.(A, A');

        W = B*B - spdiagm(diag(B*B));
        
        @assert issymmetric(W);
        return W;
    end
    #-------------------------
end
#---------------------------------------------------------------------------------
