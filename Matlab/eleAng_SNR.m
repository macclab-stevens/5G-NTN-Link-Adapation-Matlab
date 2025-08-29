
b   = Case9MCSThroughputCalulation240524212808;

figure(1);clf;
hold on
grid on 
scatter(b.eleAnge,b.SNR,1)
ylabel('SNR')
xlabel('Elevation Angle')
hold off