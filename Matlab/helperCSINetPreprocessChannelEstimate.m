function HtruncReal = helperCSINetPreprocessChannelEstimate(Hest,opt)
%helperCSINetPreprocessChannelEstimate Preprocess channel estimate
%   HDECIM = helperCSINetPreprocessChannelEstimate(HEST,OPT) preprocesses
%   the channel estimate, HEST, by decimating using 2D-FFT and 2D-IFFT. OPT
%   is the autoencoder options structure.
%
%   See also CSICompressionAutoencoderExample,
%   helperCSINetPostprocessChannelEstimate.

%   Copyright 2022 The MathWorks, Inc.

maxDelay = opt.MaxDelay;
[nSub,~,nRx,nTx] = size(Hest);

midPoint = floor(nSub/2);
lowerEdge = midPoint - (nSub-maxDelay)/2 + 1;
upperEdge = midPoint + (nSub-maxDelay)/2;

% Average over symbols (one slot)
H = squeeze(mean(Hest,2));
H = permute(H,[1 3 2]);

% Decimate over subcarriers using 2-D FFT for each Rx antenna
HtruncReal = zeros(maxDelay,nTx,2,nRx,'like',real(H(1)));
for rx = 1:nRx
  Hdft2 = fft2(H(:,:,rx));
  Htemp = Hdft2([1:lowerEdge-1 upperEdge+1:end],:);
  Htrunc = ifft2(Htemp);

  if opt.Normalization
    meanVal = opt.MeanVal;
    stdVal = opt.StdValue;
    targetSTD = opt.TargetSTDValue;
    HtruncReal(:,:,1,rx) = (real(Htrunc) - meanVal) / stdVal * targetSTD + 0.5;
    HtruncReal(:,:,2,rx) = (imag(Htrunc) - meanVal) / stdVal * targetSTD + 0.5;
  else
    HtruncReal(:,:,1,rx) = real(Htrunc);
    HtruncReal(:,:,2,rx) = imag(Htrunc);
  end
end