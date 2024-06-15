% b   = Case9MCSThroughputCalulation240524212808
% w10 = Case9MCSThroughputCalulationBLERw10Tbler0
% w50 = Case9MCSThroughputCalulationBLERw50Tbler0
% w100 = Case9MCSThroughputCalulationBLERw100Tbler0
% w200 = Case9MCSThroughputCalulationBLERw200Tbler0
% w1000 = Case9MCSThroughputCalulationBLERw1000Tbler0
w2000 = Case9MCSThroughputCalulationBLERw2000Tbler0
w2005 = Case9MCSThroughputCalulationBLERw2000Tbler1
w2010 = Case9MCSThroughputCalulationBLERw2000Tbler2

figure(1);clf;
hold on
grid on
lag = 10;
% wavg10 = movavg(w10.BLER,'simple',lag);
% wavg50 = movavg(w50.BLER,'simple',lag);
% wavg100 = movavg(w100.BLER,'simple',lag);
% wavg200 = movavg(w200.BLER,'simple',lag);
% wavg1000 = movavg(w1000.BLER,'simple',lag);
wavg2000 = movavg(w2000.BLER,'simple',lag);
wavg2005 = movavg(w2005.BLER,'simple',lag);
wavg2010 = movavg(w2010.BLER,'simple',lag);


% scatter(w10.eleAnge,wavg10,1)
% scatter(w50.eleAnge,wavg50,1)
% scatter(w100.eleAnge,wavg100,1)
% scatter(w200.eleAnge,wavg200,1)
% scatter(w1000.eleAnge,wavg1000,1)
scatter(w2000.eleAnge,wavg2000,1)
scatter(w2005.eleAnge,wavg2005,1)
scatter(w2010.eleAnge,wavg2010,1)

ylim([0 .18])
xlim([ 0 90])
legend('Window = 10','Window = 50','Window = 100','Window = 200','Window = 1000','Window = 2000')
hold off

% figure(1);clf;
% hold on
% grid on
% lag = 500;
% wavg10 = movavg(w10.RATE,'simple',lag);
% wavg50 = movavg(w50.RATE,'simple',lag);
% wavg100 = movavg(w100.RATE,'simple',lag);
% wavg200 = movavg(w200.RATE,'simple',lag);
% wavg1000 = movavg(w1000.RATE,'simple',lag);
% wavg2000 = movavg(w2000.RATE,'simple',lag);
% 
% scatter(w10.eleAnge,wavg10,1)
% scatter(w50.eleAnge,wavg50,1)
% scatter(w100.eleAnge,wavg100,1)
% scatter(w200.eleAnge,wavg200,1)
% scatter(w1000.eleAnge,wavg1000,1)
% scatter(w2000.eleAnge,wavg2000,1)
% % ylim([0 .3])
% xlim([ 0 90])
% legend('10','50','100','200','1000','2000')
% hold off

% figure(1);clf;
% hold on
% grid on
% scatter(w10.eleAnge,w10.CUMSUM_Throughput,1)
% scatter(w50.eleAnge,w50.CUMSUM_Throughput,1)
% scatter(w100.eleAnge,w100.CUMSUM_Throughput,1)
% scatter(w200.eleAnge,w200.CUMSUM_Throughput,1)
% scatter(w1000.eleAnge,w1000.CUMSUM_Throughput,1)
% scatter(w2000.eleAnge,w2000.CUMSUM_Throughput,1)
% % ylim([0 .3])
% xlim([ 0 90])
% legend('10','50','100','200','1000','2000')
% hold off


% figure(1);clf;
% hold on
% grid on
% lag = 5000;
% wavgbase = movavg(base.MCS,'simple',lag);
% wavg10 = movavg(w10.MCS,'simple',lag);
% wavg50 = movavg(w50.MCS,'simple',lag);
% wavg100 = movavg(w100.MCS,'simple',lag);
% wavg200 = movavg(w200.MCS,'simple',lag);
% wavg1000 = movavg(w1000.MCS,'simple',lag);
% wavg2000 = movavg(w2000.MCS,'simple',lag);
% 
% scatter(base.eleAnge,wavgbase,1)
% scatter(w10.eleAnge,wavg10,1)
% scatter(w50.eleAnge,wavg50,1)
% scatter(w100.eleAnge,wavg100,1)
% scatter(w200.eleAnge,wavg200,1)
% scatter(w1000.eleAnge,wavg1000,1)
% scatter(w2000.eleAnge,wavg2000,1)
% % ylim([0 .3])
% xlim([ 0 90])
% legend('base','10','50','100','200','1000','2000')
% hold off