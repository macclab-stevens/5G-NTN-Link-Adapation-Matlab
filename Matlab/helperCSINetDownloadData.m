function helperCSINetDownloadData(varargin)
%helperCSINetDownloadData Download data files
%   helperCSINetDownloadData(DISPLAYPROGRESS) downloads data files used by
%   the CSI feedback autoencoder example. DISPLAYPROGRESS (false,true)
%   controls the information text displayed by this function.

%   Copyright 2022 The MathWorks, Inc.

displayProgress = true;
if nargin == 1
    displayProgress = varargin{1};
end

dataFileName = "TrainedCSIFeedbackAutoencoder_v22b1.tar";
expFileNames = {'license.txt','csiTrainedNetwork.mat', ...
  };

url = "https://www.mathworks.com/supportfiles/spc/CSI/" ...
  + dataFileName;

dstFolder = pwd;

helperDownloadDataFile(url, ...
  dataFileName, ...
  expFileNames, ...
  dstFolder,...
  displayProgress);
end

function helperDownloadDataFile(url, archive, expFileNames, dstFolder, displayProgress)
%helperDownloadDataFile Download and uncompress data file from URL
%   helperDownloadDataFile(URL,DATAFILE,EXPFILES,DST,DISPLAYPROGRESS)
%   downloads and uncompresses DATAFILE from URL to DST folder. EXPFILES is
%   a list of expected uncompressed files. DISPLAYPROGRESS (false,true)
%   controls the information text displayed by this function.

[~, ~, fExt] = fileparts(archive);

skipExtract = true;
for p=1:length(expFileNames)
  tmpFileName = fullfile(dstFolder, expFileNames{p});
  if ~exist(tmpFileName, 'file')
    skipExtract = false;
    break
  end
end

if skipExtract && displayProgress
  disp("Files already exist. Skipping download and extract.")
else
  switch fExt
    case {'.tar', '.gz'}
      if displayProgress
        fprintf("Starting download of data files from:\n\t%s\n", url)
      end
      fileFullPath = matlab.internal.examples.downloadSupportFile('spc/CSI',...
        archive);

      if displayProgress
        disp('Download complete. Extracting files.')
      end
      untar(fileFullPath, dstFolder);

      if displayProgress
        disp("Extract complete.")
      end
 end
end
end