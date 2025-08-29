function [csiEncoder,csiDecoder,trainingInfo,autoEncOptions] = hLoadAINetwork(filename)
%hLoadAINetwork Load neural network encoder and decoder for CSI compression
%   [ENC,DEC,TINFO,ENCOPT] = hLoadAINetwork(FILENAME) loads a neural
%   network autoencoder for CSI compression specified by the MAT file name
%   FILENAME. The function returns the encoder ENC, decoder DEC, training
%   info TINFO, and autoencoder options ENCOPT.
%
%   See also helperCSINetEncode, helperCSINetDecode, helperCSINetDownloadData.

%   Copyright 2022 The MathWorks, Inc.

    % Keep a record of the network and source MAT file metadata to load the
    % network only when needed.
    persistent aiNet fileInfo csiEnc csiDec trainInfo autoEncOpt;

    % Load AI network only if the source MAT file has changed.
    finfo = dir(filename);
    if ~isequal(fileInfo,finfo)
        fileInfo = finfo;
        networkFilename = filename;
        aiNet = load(networkFilename,'net','trainInfo','options','autoEncOpt');

        % Split autoencoder into encoder and decoder.
        [csiEnc,csiDec] = helperCSINetSplitEncoderDecoder(aiNet.net,"Enc_Sigmoid");

        % Additional training info and options
        trainInfo = aiNet.trainInfo;        
        autoEncOpt = aiNet.autoEncOpt;
    end

    csiEncoder = csiEnc;
    csiDecoder = csiDec;
    trainingInfo = trainInfo;
    autoEncOptions = autoEncOpt;

end