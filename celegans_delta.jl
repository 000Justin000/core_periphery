include("testStochastic.jl")
d1 = 1.00:0.10:5.00
d2 = 0.05:0.01:0.65
omg, doe, epd, tog, tgd, omg_nev, doe_nev, epd_nev = accuracy_celegans(d1,d2)
epd_rmse = [norm(epd[i,j,:]-epd_nev)/sqrt(length(epd_nev)) for i in 1:length(d1), j in 1:length(d2)]
MAT.matwrite("celegans_delta.mat", Dict("d1"=>collect(d1), "d2"=>collect(d2), "omg"=>omg, "doe"=>doe, "epd"=>epd, "epd_rmse"=>epd_rmse, "tog"=>tog, "tgd"=>tgd))
