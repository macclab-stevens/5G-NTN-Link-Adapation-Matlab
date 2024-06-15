
function [Wprg,prgbands] = hPRGPrecoders(carrier,prgsize,Wband,bandset)
%hPRGPrecoders Get PRG precoders from precoders for arbitrary carrier RBs

%   Copyright 2023 The MathWorks, Inc.

    % Get PRG information
    prgInfo = nrPRGInfo(carrier,prgsize);

    % 'Wband' is a set of precoders for arbitrary bands in a carrier
    % resource grid, where 'bandset' is a carrier.NSizeGrid-by-1 vector
    % defining the band which applies to each carrier RB. For each PRG,
    % select the lowest band that overlaps with the PRG. Note that
    % 'bandset' can contain NaN in RBs where 'Wband' is not defined. The
    % min() operation below can encounter three situations:
    % 1. All input values are NaN, that is, the PRG doesn't overlap any 
    %    defined band. The output is NaN, that is, no precoder is defined 
    %    for that PRG
    % 2. The input values are a mixture of NaN and non-NaN values, that is,
    %    the PRG overlaps one or more undefined bands and overlaps one or
    %    more defined bands. The output is the smallest non-NaN value, that
    %    is, the selected band is the lowest defined band that overlaps the
    %    PRG
    % 3. The input values are all non-NaN, that is, the PRG overlaps with
    %    one or more defined bands. The output is the smallest value, that
    %    is, the selected band is the lowest band that overlaps the PRG
    uprg = unique(prgInfo.PRGSet);
    prgbands = arrayfun(@(x)min(bandset(prgInfo.PRGSet==x)),uprg);

    % Pad the start of 'prgbands' with NaNs for any whole PRGs that lie
    % between point A and the start of the carrier resource grid. This is
    % necessary for the correct alignment of defined PRGs in 'Wprg'
    prgbands = [NaN(prgInfo.PRGSet(1)-1,1); prgbands];

    % Get the precoders for defined PRGs and return zeros for undefined
    % PRGs
    Wprg = zeros([size(Wband,1:2) prgInfo.NPRG]);
    nin = ~isnan(prgbands);
    Wprg(:,:,nin) = Wband(:,:,prgbands(nin));

end
