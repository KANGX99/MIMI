clear ; close all; clc;
live_srcc = zeros(1,10);
live_plcc = zeros(1,10);
csiq_srcc = zeros(1,10);
csiq_plcc = zeros(1,10);
kadid10k_srcc = zeros(1,10);
kadid10k_plcc = zeros(1,10);
clive_srcc = zeros(1,10);
clive_plcc = zeros(1,10);
bid_srcc = zeros(1,10);
bid_plcc = zeros(1,10);
koniq10k_srcc = zeros(1,10);
koniq10k_plcc = zeros(1,10);

for i = 1:10
    result = load(fullfile('scores', strcat('scores',num2str(i),'.mat')));
    %live
    live_gmos = result.mos.live;
    live_pmos = result.DNN_mos.live;        
    [live_srcc(i),~,live_plcc(i),~] = verify_performance(live_gmos,live_pmos);

    %csiq
    csiq_gmos = result.mos.csiq;
    csiq_pmos = result.DNN_mos.csiq;
    [csiq_srcc(i),~,csiq_plcc(i),~] = verify_performance(csiq_gmos,csiq_pmos);


    %kadid10k
    kadid10k_gmos = result.mos.kadid10k;
    kadid10k_pmos = result.DNN_mos.kadid10k;
    [kadid10k_srcc(i),~,kadid10k_plcc(i),~] = verify_performance(kadid10k_gmos,kadid10k_pmos);


    %bid
    bid_gmos = result.mos.bid;
    bid_pmos = result.DNN_mos.bid;
    [bid_srcc(i),~,bid_plcc(i),~] = verify_performance(bid_gmos,bid_pmos);

    %clive
    clive_gmos = result.mos.clive;
    clive_pmos = result.DNN_mos.clive;    
    [clive_srcc(i),~,clive_plcc(i),~] = verify_performance(clive_gmos,clive_pmos);


    %koniq10k
    koniq10k_gmos = result.mos.koniq10k;
    koniq10k_pmos = result.DNN_mos.koniq10k;    
    [koniq10k_srcc(i),~,koniq10k_plcc(i),~] = verify_performance(koniq10k_gmos,koniq10k_pmos);

end
%median
live_srcc = median(live_srcc); live_plcc = median(live_plcc);
csiq_srcc = median(csiq_srcc); csiq_plcc = median(csiq_plcc);
kadid10k_srcc = median(kadid10k_srcc); kadid10k_plcc = median(kadid10k_plcc);
bid_srcc = median(bid_srcc); bid_plcc = median(bid_plcc);
clive_srcc = median(clive_srcc); clive_plcc = median(clive_plcc);
koniq10k_srcc = median(koniq10k_srcc); koniq10k_plcc = median(koniq10k_plcc);


weighted_srcc = live_srcc*779 + csiq_srcc*866 + kadid10k_srcc*10125 + bid_srcc*586 + clive_srcc*1162 + koniq10k_srcc*10073 ;
weighted_plcc = live_plcc*779 + csiq_plcc*866 + kadid10k_plcc*10125 + bid_plcc*586 + clive_plcc*1162 + koniq10k_plcc*10073 ;
weighted_srcc = weighted_srcc / (779+866+10125+586+1162+10073);
weighted_plcc = weighted_plcc / (779+866+10125+586+1162+10073);
