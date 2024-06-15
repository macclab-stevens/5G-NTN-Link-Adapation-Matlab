function [MCSIDX,MCSInfo] = hMCSSelect(carrier,pdsch,pdschExt,W,H,varargin)
% hMCSSelect PDSCH Modulation and coding scheme selection
%   [MCSIDX,MCSINFO] = hMCSSelect(CARRIER,PDSCH,PDSCHEXT,W,H) returns the
%   modulation and coding scheme (MCS) index MCSIDX as defined in TS 38.214
%   Section 5.1.3.1, for the specified carrier configuration CARRIER,
%   physical downlink shared channel (PDSCH) configuration PDSCH,
%   additional PDSCH configuration PDSCHEXT, MIMO precoding matrix W, and
%   estimated channel information H. The function also returns additional
%   information about the reported MCS and expected block error rate (BLER).
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
%   XOverhead       - Overhead for transport block size calculation. This
%                     field is optional and its default value is 0.
%   MCSTable        - MCS Table name specified as one of {'Table1','Table2'
%                     ,'Table3','Table4'}, as defined in TS 38.214 Section
%                     5.1.3.1. This field is optional and its default value
%                     is 'Table1'.
%   
%   W is the MIMO precoding matrix of size NumLayers-by-P-by-NPRG.
%   NumLayers is the number of layers, P is the number of reference signal
%   ports, and NPRG the number of precoding resource block group (PRGs) in
%   the BWP.
%
%   H is the channel estimation matrix of size K-by-L-by-NRxAnts-by-P,
%   where K is the number of subcarriers in the carrier resource grid, L is
%   the number of orthogonal frequency division multiplexing (OFDM) symbols
%   spanning one slot, and NRxAnts is the number of receive antennas.  
%
%   MCSIDX is a 1-by-NCW vector containing the lowest MCS indices for each
%   codeword that ensure a transport BLER below 0.1.
%   
%   MCSInfo is a structure with these fields:
%   TableRow        - 1-by-4 vector containing the MCS index, modulation 
%                     order, target code rate, and efficiency for the
%                     configured MCS table and the selected MCS.
%   TransportBLER   - 1-by-NCW vector containing the estimated transport 
%                     BLER for each codeword and the selected MCS index.
%
%   [MCSIDX,MCSINFO] = hMCSSelect(...,NVAR) specifies the estimated noise
%   variance at the receiver NVAR as a nonnegative scalar. By default, the
%   value of NVAR is 1e-10.
%
%   [MCSIDX,MCSINFO] = hMCSSelect(...,NVAR,BLERTARGET) specifies the
%   maximum BLER for the selected MCS index. The function selects the
%   lowest MCS index that ensures a BLER below the BLERTARGET value. The
%   default value is 0.1.

%   Copyright 2022 The MathWorks, Inc.

    narginchk(5,7);
    [pdsch,pdschExt,nVar,BLERTarget,NStartBWP,NSizeBWP] = validateInputs(carrier,pdsch,pdschExt,W,H,varargin{:});

    % Trim H based on BWP bandwidth and freq. position
    bottomsc = 12*(NStartBWP - carrier.NStartGrid);
    topsc = bottomsc + 12*NSizeBWP;
    H = H(1+bottomsc:topsc,:,:,:);

    % Permute to facilitate SINR calculation
    W = permute(W,[2 1 3]); % P-by-NumLayers-by-NPRG

    if size(W,3) > 1 % Subband precoding
        % To calculate the SINR per RE for that precoding matrix, map the input
        % precoder W to the appropriate REs
        K = size(H,1); % Number of subcarriers
        L = size(H,2); % Number of OFDM symbols    
        W = mapMIMOPrecoderToREs(NStartBWP,NSizeBWP,pdschExt.PRGBundleSize,W,K,L);
    end

    % Calculate SINR after MIMO precoding
    Hre = permute(H,[3 4 1 2]);
    SINRsperRE = hPrecodedSINR(Hre(:,:,:),nVar,W(:,:,:));
    
    % Find wideband MCS, effective SINR and estimated BLER per subband
    [MCSIDX,MCSTableRow,BLER] = mcsSelect(carrier,pdsch,pdschExt.XOverhead,SINRsperRE,pdschExt.MCSTable,BLERTarget);

    % Create output info
    MCSInfo = struct();
    MCSInfo.TableRow = MCSTableRow;
    MCSInfo.TransportBLER = BLER;

end

function [mcsIndex,mcsTableRow,transportBLER] = mcsSelect(carrier,pdsch,xOverhead,SINRs,MCSTableName,blerThreshold)

    % Initialize L2SM for MCS calculation
    l2sm = nr5g.internal.L2SM.initialize(carrier);

    % Initialize outputs
    ncw = pdsch.NumCodewords;
    mcsIndex = NaN(1,ncw);
    mcsTableRow = NaN(1,4);
    transportBLER = NaN(1,ncw);    

    % SINR per layer without NaN
    SINRs = 10*log10(SINRs+eps(SINRs));
    nonnan = ~any(isnan(SINRs),2);
    if ~any(nonnan,'all')
        return;
    end
    SINRs = SINRs(nonnan,:);

    % Get modulation orders and target code rates from MCS table
    mcsTable = getMCSTable(MCSTableName);

    % MCS selection
    [~,mcsIndex,mcsInfo] = nr5g.internal.L2SM.cqiSelect(l2sm,carrier,pdsch,xOverhead,SINRs,mcsTable(:,2:3),blerThreshold);
    
    % Get modulation orders and target code rates from MCS table and
    % transport BLER
    mcsTableRow = mcsTable(mcsIndex+1,:);
    transportBLER = mcsInfo.TransportBLER;

end

function WRE = mapMIMOPrecoderToREs(NStartBWP,NSizeBWP,PRGBundleSize,W,K,L)

    % Calculate the number of subbands and size of each subband for the
    % configuration.
    subbandInfo = getSubbandInfo(NStartBWP,NSizeBWP,PRGBundleSize);

    WRE = zeros([size(W,1:2) K L]);
    subbandStart = 0;
    for sb = 1:subbandInfo.NumSubbands
        % Subcarrier indices for this subband
        subbandSize = subbandInfo.SubbandSizes(sb);
        NRE = subbandSize*12;
        k = subbandStart + (1:NRE);

        % Replicate input precoder in this subband for all REs in the
        % subband
        WRE(:,:,k,:) = repmat(W(:,:,sb),1,1,NRE,L);

        % Compute the starting position of next subband
        subbandStart = subbandStart + NRE;
    end

end

function info = getSubbandInfo(nStartBWP,nSizeBWP,NSBPRB)
%   INFO = getSubbandInfo(NSTARTBWP,NSIZEBWP,NSBPRB) returns the subband
%   information.

    % Get the subband information
    if isempty(NSBPRB)
        numSubbands = 1;
        NSBPRB = nSizeBWP;
        subbandSizes = NSBPRB;
    else
        % Calculate the size of first subband
        firstSubbandSize = NSBPRB - mod(nStartBWP,NSBPRB);

        % Calculate the size of last subband
        if mod(nStartBWP + nSizeBWP,NSBPRB) ~= 0
            lastSubbandSize = mod(nStartBWP + nSizeBWP,NSBPRB);
        else
            lastSubbandSize = NSBPRB;
        end

        % Calculate the number of subbands
        numSubbands = (nSizeBWP - (firstSubbandSize + lastSubbandSize))/NSBPRB + 2;

        % Form a vector with each element representing the size of a subband
        subbandSizes = NSBPRB*ones(1,numSubbands);
        subbandSizes(1) = firstSubbandSize;
        subbandSizes(end) = lastSubbandSize;
    end
    % Place the number of subbands and subband sizes in the output
    % structure
    info.NumSubbands = numSubbands;
    info.SubbandSizes = subbandSizes;
end

function [pdsch,pdschExt,nVar,BLERTarget,NStartBWP,NSizeBWP] = validateInputs(carrier,pdsch,pdschExt,W,H,varargin)

    fcnName = 'hMCSSelect';
    
    % Validate the carrier configuration object
    validateattributes(carrier,{'nrCarrierConfig'},{'scalar'},fcnName,'CARRIER');

    % Validate dimensions of channel estimate and precoding matrix
    validateattributes(numel(size(H)),{'double'},{'>=',2,'<=',4},fcnName,'number of dimensions of H');
    K = carrier.NSizeGrid*12;
    L = carrier.SymbolsPerSlot;
    validateattributes(W,{'single','double'},{'size',[NaN size(H,4) NaN]},fcnName,'W');
    validateattributes(H,{class(H)},{'size',[K L NaN size(W,2)]},fcnName,'H');

    % Validate number of layers in W
    numLayers = size(W,1);
    if (numLayers < 1) || (numLayers > 8)
        error(['nr5g:' fcnName ':NumLayers'],'The first dimension of W must be between 1 to 8 elements inclusive.')
    end

    % Validate the PDSCH configuration object
    validateattributes(pdsch,{'nrPDSCHConfig'},{'scalar'},fcnName,'PDSCH');
    
    % Adjust PDSCH with the number of layers provided by W
    pdsch.NumLayers = size(W,1);

    % Validate NStartBWP and NSizeBWP from PDSCH or carrier
    if isempty(pdsch.NStartBWP) || isempty(pdsch.NSizeBWP)
        NStartBWP = carrier.NStartGrid;
        NSizeBWP = carrier.NSizeGrid;
    else
        if (pdsch.NStartBWP < carrier.NStartGrid)
            error(['nr5g:' fcnName ':NStartBWP'],'PDSCH NStartBWP must be >= CARRIER NStartGrid');
        end
        if (pdsch.NStartBWP + pdsch.NSizeBWP) > (carrier.NStartGrid + carrier.NSizeGrid)
            error(['nr5g:' fcnName ':BWPRB'],'The configured BWP is out of the carrier limits. PDSCH (NStartBWP + NSizeBWP) must be <= CARRIER (NStartGrid + NSizeGrid)');
        end
        NStartBWP = pdsch.NStartBWP;
        NSizeBWP = pdsch.NSizeBWP;
    end
    
     % Validate PRGBundleSize
    if isfield(pdschExt,'PRGBundleSize')
        if ~isempty(pdschExt.PRGBundleSize)
            validateattributes(pdschExt.PRGBundleSize,{'double','single'},{'scalar','integer','nonnegative','finite'},fcnName,'PRGBundleSize');
        end
    else
        pdschExt.PRGBundleSize = [];
    end

    % Validate XOverhead
    if isfield(pdschExt,'XOverhead')
        validateattributes(pdschExt.XOverhead,{'numeric'},{'scalar','integer','nonnegative'},fcnName,'XOverhead');
    else
        pdschExt.XOverhead = 0;
    end
    
    % Validate MCS table name
    if isfield('MCSTable',pdschExt)
        validatestring(pdschExt.MCSTable,{'Table1','Table2','Table3','Table4'},fcnName,'MCSTable field');
    else
        pdschExt.MCSTable = 'Table1';
    end

    % Validate noise variance
    nVar = 1e-10;
    BLERTarget = 0.1;
    if nargin > 5
        nVar = varargin{1};
        validateattributes(nVar,{'double','single'},{'scalar','real','nonnegative','finite'},fcnName,'NVAR');
        if nargin > 6
            BLERTarget = varargin{2};
            validateattributes(BLERTarget,{'single','double'},{'scalar','<',1},fcnName,'BLERTarget field');   
        end
    end
    
end

function MCSTable = getMCSTable(tableName)

    tables = MCSTables();

    tabNames = {'Table1','Table2','Table3','Table4'};
    MCSTable = tables{strcmpi(tableName,tabNames)};

end

function tablesOut = MCSTables()

    persistent tables;

    if isempty(tables)

        % TS 38.214 Table 5.1.3.1-1
        tables{1} = [...
                    0   2 120 0.2344
                    1   2 157 0.3066
                    2   2 193 0.3770
                    3   2 251 0.4902
                    4   2 308 0.6016
                    5   2 379 0.7402
                    6   2 449 0.8770
                    7   2 526 1.0273
                    8   2 602 1.1758
                    9   2 679 1.3262
                    10  4 340 1.3281
                    11  4 378 1.4766
                    12  4 434 1.6953
                    13  4 490 1.9141
                    14  4 553 2.1602
                    15  4 616 2.4063
                    16  4 658 2.5703
                    17  6 438 2.5664
                    18  6 466 2.7305
                    19  6 517 3.0293
                    20  6 567 3.3223
                    21  6 616 3.6094
                    22  6 666 3.9023
                    23  6 719 4.2129
                    24  6 772 4.5234
                    25  6 822 4.8164
                    26  6 873 5.1152
                    27  6 910 5.3320
                    28  6 948 5.5547];
        
        % Table 5.1.3.1-2
        tables{2} = [...
                    0   2    120    0.2344
                    1   2    193    0.3770
                    2   2    308    0.6016
                    3   2    449    0.8770
                    4   2    602    1.1758
                    5   4    378    1.4766
                    6   4    434    1.6953
                    7   4    490    1.9141
                    8   4    553    2.1602
                    9   4    616    2.4063
                    10  4    658    2.5703
                    11  6    466    2.7305
                    12  6    517    3.0293
                    13  6    567    3.3223
                    14  6    616    3.6094
                    15  6    666    3.9023
                    16  6    719    4.2129
                    17  6    772    4.5234
                    18  6    822    4.8164
                    19  6    873    5.1152
                    20  8    682.5  5.3320
                    21  8    711    5.5547
                    22  8    754    5.8906
                    23  8    797    6.2266
                    24  8    841    6.5703
                    25  8    885    6.9141
                    26  8    916.5  7.1602
                    27  8    948    7.4063];
        
        % Table 5.1.3.1-3
        tables{3} = [...
                    0   2 30 0.0586
                    1   2 40 0.0781
                    2   2 50 0.0977
                    3   2 64 0.1250
                    4   2 78 0.1523
                    5   2 99 0.1934
                    6   2 120 0.2344
                    7   2 157 0.3066
                    8   2 193 0.3770
                    9   2 251 0.4902
                    10  2 308 0.6016
                    11  2 379 0.7402
                    12  2 449 0.8770
                    13  2 526 1.0273
                    14  2 602 1.1758
                    15  4 340 1.3281
                    16  4 378 1.4766
                    17  4 434 1.6953
                    18  4 490 1.9141
                    19  4 553 2.1602
                    20  4 616 2.4063
                    21  6 438 2.5664
                    22  6 466 2.7305
                    23  6 517 3.0293
                    24  6 567 3.3223
                    25  6 616 3.6094
                    26  6 666 3.9023
                    27  6 719 4.2129
                    28  6 772 4.5234];
        
        % Table 5.1.3.1-4
        tables{4} = [...
                    0 2 120 0.2344
                    1 2 193 0.3770
                    2 2 449 0.8770
                    3 4 378 1.4766
                    4 4 490 1.9141
                    5 4 616 2.4063
                    6 6 466 2.7305
                    7 6 517 3.0293
                    8 6 567 3.3223
                    9 6 616 3.6094
                    10 6 666 3.9023
                    11 6 719 4.2129
                    12 6 772 4.5234
                    13 6 822 4.8164
                    14 6 873 5.1152
                    15 8 682.5 5.3320
                    16 8 711 5.5547
                    17 8 754 5.8906
                    18 8 797 6.2266
                    19 8 841 6.5703
                    20 8 885 6.9141
                    21 8 916.5 7.1602
                    22 8 948 7.4063
                    23 10 805.5 7.8662
                    24 10 853 8.3301
                    25 10 900.5 8.7939
                    26 10 948 9.2578];
    end

    tablesOut = tables;

end