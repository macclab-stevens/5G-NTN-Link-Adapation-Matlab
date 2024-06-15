function Hhat = helperCSINetPostprocessChannelEstimate(HtruncReal,opt)
%helperCSINetPostprocessChannelEstimate Postprocess channel estimate
%   HHAT = helperCSINetPostprocessChannelEstimate(HDECIM,OPT) postprocesses
%   the decimated channel estimates, HDECIM, by interpolation using
%   2D-FFT and 2D-IFFT. OPT is the autoencoder options structure.
%
%   See also CSICompressionAutoencoderExample,
%   helperCSINetPreprocessChannelEstimate.

%   Copyright 2022 The MathWorks, Inc.

maxDelay = opt.MaxDelay;
meanVal = opt.MeanVal;
stdVal = opt.StdValue;
targetSTD = opt.TargetSTDValue;
nSub = opt.NumSubcarriers;
nTx = size(HtruncReal,2);

midPoint = floor(nSub/2);
lowerEdge = midPoint - (nSub-maxDelay)/2 + 1;

Htemp = (HtruncReal(:,:,:) - 0.5) / targetSTD * stdVal + meanVal;
Htrunc = Htemp(:,:,1) + 1i*Htemp(:,:,2);
HdaTrunc = fft2(Htrunc);
Hda = [HdaTrunc(1:lowerEdge-1,:); zeros((nSub-maxDelay),nTx); HdaTrunc(lowerEdge:end,:)];
Hhat = ifft2(Hda);

