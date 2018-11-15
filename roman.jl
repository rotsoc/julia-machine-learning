struct Roman
    n::Int
end

num_map = [1000=>"M", 900=>"CM", 500=>"D", 400=>"CD", 100=>"C",
 90=>"XC", 50=>"L", 40=>"XL", 10=>"X", 9=>"IX", 5=>"V",
 4=>"IV", 1=>"I"]

function romanstr(n, i=1)
    if n==0
        return ""
    end
    if n>=5000
        return(string(n))
    end
    if n>=first(num_map[i])
        string(last(num_map[i]), romanstr(n-first(num_map[i]), i))
    else
        romanstr(n,i+1)
    end
end

Base.show(io::IO, r::Roman) = print(io,romanstr(r.n))

import Base: +, *, ^

+(a::Roman,b::Roman)=Roman(a.n+b.n)
*(i::Roman,j::Roman)=Roman(i.n*j.n)
^(a,n)=prod(fill(a,n))
