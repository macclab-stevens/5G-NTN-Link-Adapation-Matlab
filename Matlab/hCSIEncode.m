function csiReport = hCSIEncode(carrier,csirs,H,nVar,alg)
%hCSIEncode Encode channel state information (CSI)
%   CSIREPORT = hCSIEncode(CARRIER,CSIRS,H,NVAR,ALG) returns the channel
%   state information report CSIREPORT corresponding to the channel
%   estimates H, noise estimate NVAR, and the selected algorithmic
%   configuration ALG.
%
%   CARRIER is a carrier configuration object, as described in <a
%   href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>.
%
%   CSIRS is a CSI-RS configuration object, as described in <a href="matlab:help('nrCSIRSConfig')">nrCSIRSConfig</a>.
%
%   H is the channel estimation matrix of size K-by-L-by-nRxAnts-by-P,
%   where K is the number of subcarriers in the carrier resource grid, L is
%   the number of OFDM symbols spanning one slot, nRxAnts is the number of
%   receive antennas, and P is the number of CSI-RS antenna ports.
%
%   NVAR is the estimated noise variance at the receiver specified as a
%   nonnegative scalar.
%
%   ALG is an algorithmic configuration structure with these fields:
%   CSIReportMode   - 'RI-PMI-CQI','AI CSI compression','Perfect CSI'. 
%                     For more information about these modes, see the
%                     description of the CSIREPORT output.
%   These fieds only apply when CSIReportMode = 'RI-PMI-CQI':
%   CSIReportConfig - Structure containing the CSI report configuration.
%                     For more information, see the REPORTCONFIG input of 
%                     <a href="matlab:help('hRISelect')">hRISelect</a>, <a href="matlab:help('hDLPMISelect')">hDLPMISelect</a>, and <a href="matlab:help('hCQISelect')">hCQISelect</a>.
%   PerfectChannelEstimator - (Optional) true or false. Default true.
%   This field only applies when CSIReportMode = 'AI CSI compression':
%   NetworkFilename - Name of the file containing the neural network autoencoder.
%
%   CSIREPORT is a structure containing the CSI report. 
%   When ALG.CSIReportMode = 'RI-PMI-CQI', CSIREPORT contains these fields:
%   RI      - Rank indicator
%   PMI     - Precoding matrix indicator
%   CQI     - Channel quality indicator
%   W       - Pecoding matrix associated with the reported PMI.
%   NSlot   - Number of the slot in which the report was generated.
%   For more information about RI, PMI, and CQI, see the outputs of <a href="matlab:help('hRISelect')">hRISelect</a>, 
%   <a href="matlab:help('hDLPMISelect')">hDLPMISelect</a>, and <a href="matlab:help('hCQISelect')">hCQISelect</a>.
%   When ALG.CSIReportMode = 'AI CSI compression', CSIREPORT contains 
%   these fields:
%   H       - Compressed channel estimates returned by the neural network
%             autoencoder. The contents of this field correspond to the
%             output CW of <a href="matlab:help('helperCSINetEncode')">helperCSINetEncode</a>.
%   NVAR    - Noise variance estimate.
%   NSlot   - Number of the slot in which the report was generated.
%   When ALG.CSIReportMode = 'Perfect CSI', CSIREPORT contains these fields:
%   H       - Input channel estimate matrix H.
%   NVAR    - Noise variance estimate.
%   NSlot   - Number of the slot in which the report was generated.
%
%   See also hCSIDecode, helperCSINetEncode, hRISelect, hDLPMISelect, hCQISelect

%   Copyright 2022 The MathWorks, Inc.

    % Load CSI encoder and decoder if AI CSI compression mode is enabled
    if alg.CSIReportMode == "AI CSI compression" 
        [csiEncoder,csiDecoder,~,autoEncOptions] = hLoadAINetwork(alg.AINetworkFilename);
    end

    % Generate CSI report from channel estimates.
    if alg.CSIReportMode == "RI-PMI-CQI"
        % In 5G, the CSI report includes the rank indicator (RI), precoding
        % matrix indicator (PMI), and channel quality indicator (CQI).
        csiReport = hCSISelect(carrier,csirs,alg.CSIReportConfig,H,nVar,alg);
    else
        % In Perfect CSI mode, the report includes the full channel estimate
        if alg.CSIReportMode == "Perfect CSI"
            Hc = H;
        else
            % Encode channel estimates using an AI network
            Hc = helperCSINetEncode(csiEncoder,H,autoEncOptions);
        end
        % Create CSI report
        csiReport = struct();
        csiReport.H = Hc;
        csiReport.nVar = nVar;
        csiReport.NSlot = carrier.NSlot;
    end
    
end

% Select RI, PMI, and CQI for the input channel and noise estimates
function csi = hCSISelect(carrier,csirs,reportConfig,H,nVar,alg)

    if isfield(alg,'PerfectChannelEstimator') && ~alg.PerfectChannelEstimator
        nVar = nVar*1.602;
    end
    
    % Calculate the RI value using channel estimate
    rank = hRISelect(carrier,csirs,reportConfig,H,nVar,'MaxSE');

    % If no rank is allowed, use rank 1 by default.
    if isnan(rank)
        rank = 1;
    end

    % Select CQI conditioned to that MIMO precoder
    [cqi,pmi,~,pmiInfo] = hCQISelect(carrier,csirs,reportConfig,rank,H,nVar);

    % Collect CSI
    csi = struct();
    csi.RI = rank;
    csi.CQI = cqi;
    csi.PMI = pmi;
    csi.W = pmiInfo.W;
    csi.NSlot = carrier.NSlot;

end
