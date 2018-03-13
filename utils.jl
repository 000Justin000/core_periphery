function join(a, b)
    l = [];
    for element in a
        push!(l, element);
    end

    for element in b
        push!(l, element);
    end
    
    filter!(element->element != 0.0, l);

    l = unique(l);

    if (length(l) == 0)
        l = 0.0;
    elseif (length(l) == 1)
        l = l[1];
    end

    return l;
end

function merge_node(A, a, b)
    # merge direction a <- b
    @assert size(A,1) == size(A,2);
    
    n = size(A,1);

    for i in 1:n
        if (i != a)
            A[a,i] = join(A[a,i], A[b,i]);
            A[i,a] = join(A[i,a], A[i,b]);
        end
    end

    select = trues(n);
    select[b] = false;

    return A[select,select];
end
