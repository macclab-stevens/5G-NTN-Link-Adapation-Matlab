function codeword = helperCSINetEncode(encNet, Hest, opt)
%helperCSINetEncode Compress channel estimates with encoder neural network
%   CW = helperCSINetEncode(ENC,HEST,OPT) compresses channel estimates,
%   HEST, using the encoder neural network, ENC. OPT is the autoencoder
%   options structure.
%
%   See also CSICompressionAutoencoderExample, helperCSINetDecode.

%   Copyright 2022 The MathWorks, Inc.

[nSub,nSym,nRx,nTx,N] = size(Hest);

assert(nSub == opt.NumSubcarriers, ...
    sprintf("Number of subcarriers (%d) in Hest does not match autoencoder's expected number of subcarriers (%d).",nSub,opt.NumSubcarriers))
assert(nSym == opt.NumSymbols, ...
    sprintf("Number of symbols (%d) in Hest does not match autoencoder's expected number of symbols (%d).",nSym,opt.NumSymbols))
assert(nTx == opt.NumTxAntennas, ...
    sprintf("Number of Tx antennas (%d) in Hest does not match autoencoder's expected number of Tx antennas (%d).",nTx,opt.NumTxAntennas))

dummy = predict(encNet,zeros(opt.MaxDelay,nTx,2),'ExecutionEnvironment','cpu');
codeword = zeros(nRx,N,length(dummy),'single');
for n=1:N
  HtruncReal = helperCSINetPreprocessChannelEstimate(Hest(:,:,:,:,n),opt);
  codeword(:,n,:) = predict(encNet,HtruncReal,...
    'ExecutionEnvironment','cpu');
end
end

