b   = Case9MCSThroughputCalulation240524212808;
w10 = Case9MCSThroughputCalulationBLERw10Tbler0;
w50 = Case9MCSThroughputCalulationBLERw50Tbler0;
w100 = Case9MCSThroughputCalulationBLERw100Tbler0;
w200 = Case9MCSThroughputCalulationBLERw200Tbler0;
w1000 = Case9MCSThroughputCalulationBLERw1000Tbler0;
w2000 = Case9MCSThroughputCalulationBLERw2000Tbler0
% w2005 = Case9MCSThroughputCalulationBLERw2000Tbler1;
% w2010 = Case9MCSThroughputCalulationBLERw2000Tbler2;

grid on
lag = 2000;
colororder({'k','k'})
yyaxis left
hold on
wavgb = movavg(b.BLER,'simple',lag);
wavg10 = movavg(w10.BLER,'simple',lag);
wavg50 = movavg(w50.BLER,'simple',lag);
wavg100 = movavg(w100.BLER,'simple',lag);
wavg200 = movavg(w200.BLER,'simple',lag);
wavg1000 = movavg(w1000.BLER,'simple',lag);
wavg2000 = movavg(w2000.BLER,'simple',lag);
% wavg2005 = movavg(w2005.BLER,'simple',lag);
% wavg2010 = movavg(w2010.BLER,'simple',lag);

scatter(b.eleAnge,wavgb,1,[0 0.4470 0.7410])
scatter(w10.eleAnge,wavg10,1,[0.8500 0.3250 0.0980])
scatter(w50.eleAnge,wavg50,1,[0.9290 0.6940 0.1250])
scatter(w100.eleAnge,wavg100,1,[0.4940 0.1840 0.5560])
scatter(w200.eleAnge,wavg200,1,[0.4660 0.6740 0.1880])
scatter(w1000.eleAnge,wavg1000,1,[0.3010 0.7450 0.9330])
scatter(w2000.eleAnge,wavg2000,1,[0.6350 0.0780 0.1840])
% scatter(w2005.eleAnge,wavg2005,1,[0 0.4470 0.7410])
% scatter(w2010.eleAnge,wavg2010,1,[0.4940 0.1840 0.5560])

ylim([0 .3])
xlim([0.5 90])
ylabel("Avg BLER")
xlabel("Elevation Angle")
hold off
hold on
yyaxis right 
lag = 5000;

ravgb = movavg(b.RATE,'simple',lag);
ravg10 = movavg(w10.RATE,'simple',lag);
ravg50 = movavg(w50.RATE,'simple',lag);
ravg100 = movavg(w100.RATE,'simple',lag);
ravg200 = movavg(w200.RATE,'simple',lag);
ravg1000 = movavg(w1000.RATE,'simple',lag);
ravg2000 = movavg(w2000.RATE,'simple',lag);
% ravg2005 = movavg(w2005.RATE,'simple',lag);
% ravg2010 = movavg(w2010.RATE,'simple',lag);

scatter(b.eleAnge,ravgb,1,[0 0.4470 0.7410])
scatter(w10.eleAnge,ravg10,1,[0.8500 0.3250 0.0980])
scatter(w50.eleAnge,ravg50,1,[0.9290 0.6940 0.1250])
scatter(w100.eleAnge,ravg100,1,[0.4940 0.1840 0.5560])
scatter(w200.eleAnge,ravg200,1,[0.4660 0.6740 0.1880])
scatter(w1000.eleAnge,ravg1000,1,[0.3010 0.7450 0.9330])
scatter(w2000.eleAnge,ravg2000,1,[0.6350 0.0780 0.1840])
% % scatter(w2005.eleAnge,ravg2005,1,[0 0.4470 0.7410])
% % scatter(w2010.eleAnge,ravg2010,1,[0.4940 0.1840 0.5560])

% scatter(w10.eleAnge,w10.CUMSUM_Throughput,1,[0 0.4470 0.7410])
% scatter(w50.eleAnge,w50.CUMSUM_Throughput,1,[0.8500 0.3250 0.0980])
% scatter(w100.eleAnge,w100.CUMSUM_Throughput,1,[0.9290 0.6940 0.1250])
% scatter(w200.eleAnge,w200.CUMSUM_Throughput,1,[0.4940 0.1840 0.5560])
% scatter(w1000.eleAnge,w1000.CUMSUM_Throughput,1,[0.4660 0.6740 0.1880])
% scatter(w2000.eleAnge,w2000.CUMSUM_Throughput,1,[0.6350 0.0780 0.1840])

ylim([-5e6,12e6])
ylabel("Avg Rate (bps)")
hold off

lgd = legend('Baseline','Window = 10','Window = 50','Window = 100','Window = 200','Window = 1000','Window = 2000')
lgd.Location = "northwest"
lgd.Title.String = "Target BLER = 1%"
