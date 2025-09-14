function [mod,Qm,tcr,mcs,wtx] = MCS_Allo_Algo1(carrier,pdsch,pdschExt,csiReport,alg,BLER,TargetBLER,cur_mcs,snr_val)
                                                % carrier,pdsch,pdschextra,csiReports(repIdx),csiFeedbackOpts,BLER,TargetBLER,cur_mcs
  
%hCSIDecode Decode channel state information (CSI) report
%   [MOD,TCR,WTX] = hCSIDecode(CARRIER,PDSCH,PDSCHEXT,CSIREPORT,ALG)
%   returns the modulation MOD, target code rate TCR, and MIMO precoder
%   matrix WTX selected for the input carrier configuration CARRIER,
%   physical downlink shared channel (PDSCH) configuration PDSCH,
%   additional PDSCH configuration PDSCHEXT, CSI report CSIREPORT, and the
%   algorithmic configuration ALG.
%
%   CARRIER is a carrier configuration object, as described in <a href="matlab:help('nrCarrierConfig')">nrCarrierConfig</a>.
%
%   PDSCH is a PDSCH configuration object, as described in <a href="matlab:help('nrPDSCHConfig')">nrPDSCHConfig</a>. 
%
%   PDSCHEXT is an extended PDSCH configuration structure with these fields:
%   PRGBundleSize   - Number of consecutive RBs with the same MIMO precoder,
%                     as defined in TS 38.214 Section 5.1.2.3. It can be
%                     [], 2, or 4. Use [] to indicate wideband. This field
%                     is optional and its default value is [].
%   MCSTable        - Modulation and coding scheme table name specified as
%                     one of {'Table1','Table2','Table3','Table4'}, as
%                     defined in TS 38.214 Section 5.1.3.1. This field is
%                     optional and its default value is 'Table1'.
%
%   ALG is a structure with these fields:
%   CSIReportMode   - 'RI-PMI-CQI','AI CSI compression','Perfect CSI'. 
%                     For more information about these modes, see the
%                     description of the CSIREPORT input below.
%   These fieds only apply when CSIReportMode = 'RI-PMI-CQI':
%   CSIReportConfig - Structure containing the CSI report configuration.
%                     For more information, see the REPORTCONFIG input in 
%                     <a href="matlab:help('hRISelect')">hRISelect</a>, <a href="matlab:help('hDLPMISelect')">hDLPMISelect</a>, and <a href="matlab:help('hCQISelect')">hCQISelect</a>.
%                     The function uses this information to map the input 
%                     CQI to PDSCH MCS and subband MIMO precoder to PDSCH PRG.
%   These fields only apply when CSIReportMode = 'AI CSI compression':
%   NetworkFilename          - AI network file name
%   PerfectChannelEstimator  - (Optional) true or false. Default true.
%
%   CSIREPORT is a structure containing the CSI report provided by <a href="matlab:help('hCSIEncode')">hCSIEncode</a>.
%   When ALG.CSIReportMode = 'RI-PMI-CQI', CSIREPORT must contain these fields:
%   RI      - Rank indicator
%   CQI     - Channel quality indicator
%   W       - Pecoding matrix associated with the reported precoding matrix
%             indicator (PMI).
%   For more information about RI, PMI, and CQI, see the outputs of <a href="matlab:help('hRISelect')">hRISelect</a>, 
%   <a href="matlab:help('hDLPMISelect')">hDLPMISelect</a>, and <a href="matlab:help('hCQISelect')">hCQISelect</a>.
%   When ALG.CSIReportMode = 'AI CSI compression', CSIREPORT must contain 
%   these fields:
%   H       - Compressed channel estimates returned by the neural network
%             autoencoder. The contents of this field correspond to the
%             output CSICW of <a href="matlab:help('helperCSINetEncode')">helperCSINetEncode</a>.
%   NVAR    - Noise variance estimate.
%   When ALG.CSIReportMode = 'Perfect CSI', CSIREPORT must contain these fields:
%   H       - Channel estimate matrix.
%   NVAR    - Noise variance estimate.
%
%   See also hCSIEncode, helperCSINetDecode, hRISelect, hDLPMISelect, hCQISelect

%   Copyright 2022-2023 The MathWorks, Inc.

    % if alg.CSIReportMode == "AI CSI compression" 
    %     [csiEncoder,csiDecoder,~,autoEncOptions] = hLoadAINetwork(alg.AINetworkFilename);
    % end
    lut = readtable('snr_cqi_lut_dense_v2.csv'); % Assumes headers: snr, cqi, mcs
    if alg.CSIReportMode == "RI-PMI-CQI"
        csiReportConfig = alg.CSIReportConfig;
        % disp("HI")
        % snr_val
        snr_val = round(snr_val, 1);
        % lut.cqi
        % csiReport.CQI
        
        row = lut(lut.snr == snr_val & lut.cqi == csiReport.CQI, :);
        if ~isempty(row)
            disp(sprintf('SNR:%f,CQI:%f,NewCQI:%f\n', snr_val, csiReport.CQI, row.mcs));
            csiReport.CQI = row.mcs;
            if csiReport.CQI>15
                csiReport.CQI=15
            end
        else
            csiReport.CQI = 0; % or handle "not found" case
        end

        % % Map CSI to modulation and target code rate
        % csiReportConfig = alg.CSIReportConfig;
        % curCQI = csiReport.CQI;
        % NewMCS = 0;
        % % fprintf("\n\t\t MCS_Allo_Algo1 curCQI:%2f curMCS:%2f BLER:%.4f TargetBler:%.2f",curCQI,cur_mcs,BLER,TargetBLER)
        % %if the reported CQI is > our current MCS and we're under our
        % %target increase the MCS by the new report. 
        % if curCQI > cur_mcs && BLER < TargetBLER
        %     % fprintf('!!')
        %     csiReport.CQI = curCQI;
        %     % fprintf('!!%f %f',csiReport.CQI,cur_mcs)
        % %if the reported CQI is < lower than cur MCS but we're still under
        % %our target BLER do NOT decrease the MCS . leave it the same. 
        % elseif curCQI < cur_mcs && BLER < TargetBLER
        %     csiReport.CQI = cur_mcs;
        %     % fprintf('!!%f %f',csiReport.CQI,cur_mcs)
        % elseif BLER > TargetBLER
        %     csiReport.CQI = curCQI;
        % elseif BLER == 0
        %     csiReport.CQI = curCQI + 1;
        % 
        % end
        [mod,tcr,mcs,Qm] = CSI2MCS(csiReportConfig.CQITable,csiReport);
        
        % mcs_idx = mcs(:,1);
        % Map codebook-based precoding matrices from subbands to PRGs
        wtx = hPMISubandToPRGPrecodingMatrix(carrier,pdschExt.PRGBundleSize,csiReportConfig,csiReport.W);

    % else
    % 
    %     if alg.CSIReportMode == "Perfect CSI"
    %         H = csiReport.H;
    %     else
    % 
    %         % Decode CSI report
    %         H = helperCSINetDecode(csiDecoder,csiReport.H,autoEncOptions);
    % 
    %         % Replicate channel matrix in the time domain to span 1
    %         % slot worth of symbols
    %         H = repelem(permute(H,[1 4 2 3]),1,carrier.SymbolsPerSlot,1,1,1);
    %     end
    % 
    %     [mod,tcr,wtx] = hPDSCHTransmissionParameters(carrier,pdsch,pdschExt,H,csiReport.nVar,alg);
    % 
    end

end

% Map CQI to PDSCH MCS for the specified CQI table
function [mod,tcr,mcs,Qm] = CSI2MCS(cqiTable,csiReport)
    
    % Configure MCS based on CQI
    cqi = csiReport.CQI(1,:); % Wideband CQI
    ncw = ceil(csiReport.RI/4);
    cqi = max([ones(1,ncw); cqi],[],1); % map CQI 0 -> CQI 1
    mcs = hCQITables(cqiTable,cqi);
    Qm = mcs(:,2).';
    tcr = mcs(:,3).'/1024;
    modLists = repmat({'QPSK','16QAM','64QAM','256QAM'}.',1,ncw);
    mod = modLists(Qm==repmat([2 4 6 8].',1,ncw));

end

% Map codebook-based precoding matrices from subbands to PRGs
function newWtx = hPMISubandToPRGPrecodingMatrix(carrier,prgbundlesize,reportConfig,W)

    subbandInfo = hDLPMISubbandInfo(carrier,reportConfig);
    newWtx = hPRGPrecoders(carrier,prgbundlesize,W,subbandInfo.SubbandSet.');
    newWtx = permute(newWtx,[2 1 3]);

end

function [modulation,tcr,wtx] = hPDSCHTransmissionParameters(carrier,pdsch,pdschExt,Hest,nVar,alg)
% Calculate transmission parameters: MCS and MIMO precoder

    if isfield(alg,'PerfectChannelEstimator') && ~alg.PerfectChannelEstimator
        nVar = nVar*1.602;
    end

    % For each valid rank, compute the SINR after MIMO precoding. Select
    % the MCS index based on the SINRs for each rank.
    maxRank = min(size(Hest,[3 4]));
    validRanks = 1:maxRank;
    W = cell(maxRank,1);

    efficiency = NaN(maxRank,1);
    mcsRank = NaN(maxRank,2);
    for rank = validRanks

        % Select MIMO precoder from SVD of the channel estimate and
        % calculate the associated SINR
        w = hSVDPrecodingMatrix(carrier,rank,pdsch.PRBSet,Hest,nVar,pdschExt.PRGBundleSize);

        % Store MIMO precoding matrix for this rank
        W{rank} = w;

        % MCS
        pdsch.NumLayers = rank;
        pdschExt.XOverhead = 0;
        pdsch.ReservedRE = [];
        [mcs,mcsInfo(rank)] = hMCSSelect(carrier,pdsch,pdschExt,w,Hest,nVar); %#ok<AGROW>

        % Get wideband MCS index
        mcsWideband = mcs(1,:);

        ncw = length(mcsWideband);
        mcsRank(rank,:) = repmat(mcs(1,:),1,1+(ncw==1));

        % If the wideband MCS index is appropriate, calculate the efficiency
        if all(mcsWideband ~= 0)
            if ~any(isnan(mcsWideband))
                % Calculate throughput-related metric using number of
                % layers, code rate and modulation, and estimated BLER
                blerWideband = mcsInfo(rank).TransportBLER(1,:);
                ncw = numel(mcsWideband);
                cwLayers = floor((rank + (0:ncw-1)) / ncw);
                mcs = mcsInfo(rank).TableRow;
                eff = cwLayers .* (1 - blerWideband) * mcs(:,4);
                efficiency(rank) = eff;
            end
        else
            efficiency(rank) = 0;
        end

    end

    % Choose the rank that maximizes throughput.
    [~,rank] = max(efficiency);

    % Update the number of PDSCH layers based on selected rank
    numLayers = rank;

    % Configure MCS based on MCS index
    ncw = ceil(numLayers/4);
    mcs = mcsInfo(rank).TableRow;
    Qm = mcs(:,2).';
    tcr = mcs(:,3).'/1024;
    modLists = repmat({'QPSK','16QAM','64QAM','256QAM'}.',1,ncw);
    modulation = modLists(Qm==repmat([2 4 6 8].',1,ncw));    

    wtx = W{rank};

end

function [wtx,sinr] = hSVDPrecodingMatrix(carrier,numLayers,prbset,hestGrid,nVar,prgbundlesize)
% Calculate precoding matrices for all PRGs in the carrier that overlap
% with the PDSCH allocation

    persistent pdsch;
    if (isempty(pdsch))
        pdsch = nrPDSCHConfig;
    end

    pdsch.NumLayers = numLayers;
    pdsch.PRBSet = prbset;
    [wtx,hest] = hSVDPrecoders(carrier,pdsch,hestGrid,prgbundlesize);

    NPRG = size(wtx,3);
    sinr = zeros(numLayers,NPRG);
    for i = 1:NPRG
        sinr(:,i) = hPrecodedSINR(hest(:,:,i),nVar,wtx(:,:,i).');
    end

end
