

[par, fval, flag] = fminsearch(@(mypar)to_max(mypar(1), mypar(2)), [10 100] )

function val = to_max(val1, val2)
    val = 1 + val1^2 + val2^2 ;
end 