function Hhat = helperCSINetDecode(decNet, codeword, opt)
%helperCSINetDecode Decode compressed channel estimates
%   HHAT = helperCSINetDecode(DEC,CSICW,OPT) decodes compressed channel
%   estimates, CSICW, using the decoder neural network, DEC. OPT is the
%   autoencoder options structure.
%
%   See also CSICompressionAutoencoderExample, helperCSINetEncode.

%   Copyright 2022 The MathWorks, Inc.

[nRx,N,codewordLen] = size(codeword);
dummyOut = predict(decNet,zeros(1,codewordLen));
[nDelay,nTx,~] = size(dummyOut);

nSub = opt.NumSubcarriers;
assert(nTx == opt.NumTxAntennas, ...
    sprintf("Number of Tx antennas (%d) in Hest does not match autoencoder's expected number of Tx antennas (%d).",nTx,opt.NumTxAntennas))
assert(nDelay == opt.MaxDelay, ...
    sprintf("Maximum delay (%d) in network does not match autoencoder's expected maximum delay (%d).",nDelay,opt.MaxDelay))


Hhat = zeros(nSub,nRx,nTx,N,'like',codeword);
HtruncRealHat = complex(zeros(opt.MaxDelay,nTx,2,nRx*N));
for rxAnt = 1:nRx
    for n = 1:N
        HtruncRealHat(:,:,:,(n-1)*nRx+rxAnt) = predict(decNet,reshape(codeword(rxAnt,n,:),1,codewordLen),'ExecutionEnvironment','cpu');
        Hhat(:,rxAnt,:,n) = helperCSINetPostprocessChannelEstimate(HtruncRealHat(:,:,:,(n-1)*nRx+rxAnt),opt);
    end
end
end

