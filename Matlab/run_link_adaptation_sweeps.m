function run_link_adaptation_sweeps(varargin)
%RUN_LINK_ADAPTATION_SWEEPS Parallel dataset generator for NTN link adaptation.
%
% This orchestrator sweeps scenario parameters, uses all local CPU cores via
% parfor, and invokes a parameterized copy of
% NewRadioPDSCHThroughputWithCSIFeedbackExampleCode.m to dump per‑slot CSVs.
% It does not modify your original example file; instead, it writes a temporary
% copy per scenario with the desired values substituted and runs that.
%
% Usage (from Matlab/ working directory or repo root with path added):
%   run_link_adaptation_sweeps();
%   run_link_adaptation_sweeps('OutputDir','ML/data/generated', ...
%                              'SNRRange',-15:2:25, ...
%                              'Frames',8000, ...
%                              'TargetBler',[0.01 0.05], ...
%                              'BlerWindow',[200 1000 2000], ...
%                              'Altitude',[600e3 1200e3], ...
%                              'Seeds',1:6);
%
% Name/value params (defaults in parens):
%   'OutputDir'  (fullfile(pwd,'ML','data','generated'))
%   'SNRRange'   (-12:3:21)
%   'Frames'     (8000)   % frames per scenario (~80k slots at 15kHz numerology)
%   'TargetBler' ([0.01])
%   'BlerWindow' ([1000])
%   'Altitude'   ([600e3])
%   'Seeds'      (1:4)
%   'Workers'    ([] => all local cores)
%
% Notes
% - This orchestrator uses the original example’s logic unchanged. GPU use in the
%   inner loop would require edits inside the example (casting arrays to gpuArray).
% - Each scenario writes CSVs into a unique folder under OutputDir.

%% Parse arguments
p = inputParser;
p.addParameter('OutputDir', fullfile(pwd,'ML','data','generated'), @isCharLike);
p.addParameter('SNRRange', -12:3:21, @(x)isnumeric(x)&&isvector(x));
p.addParameter('Frames', 8000, @(x)isnumeric(x)&&isscalar(x));
p.addParameter('TargetBler', 0.01, @(x)isnumeric(x)&&isvector(x));
p.addParameter('BlerWindow', 1000, @(x)isnumeric(x)&&isvector(x));
p.addParameter('Altitude', 600e3, @(x)isnumeric(x)&&isvector(x));
p.addParameter('Seeds', 1:4, @(x)isnumeric(x)&&isvector(x));
p.addParameter('Workers', [], @(x)isnumeric(x)||isempty(x));
p.addParameter('UseParallel', true, @islogical);
p.addParameter('PoolType', 'Processes', @isCharLike); % 'Processes' or 'Threads'
p.addParameter('Quick', false, @islogical);
p.addParameter('LogEveryNSlots', 1000, @(x)isnumeric(x)&&isscalar(x)&&x>=1);
p.parse(varargin{:});
cfg = p.Results;

% Normalize OutputDir to an absolute, canonical path to avoid relative path
% issues inside parfor workers and nested cd calls.
cfg.OutputDir = makeAbsolutePath(cfg.OutputDir);

% Quick mode for faster iteration
if cfg.Quick
    % Trim to a tiny sweep that completes quickly
    cfg.SNRRange = unique([cfg.SNRRange(1), cfg.SNRRange(end)]); % two points
    cfg.Frames   = min(cfg.Frames, 1000);                        % <= 1000 frames
    if numel(cfg.Seeds) > 2, cfg.Seeds = cfg.Seeds(1:2); end     % two seeds
    % Default to serial execution in quick mode to avoid pool startup costs
    cfg.UseParallel = false;
end

exampleFile = fullfile(fileparts(mfilename('fullpath')),'NewRadioPDSCHThroughputWithCSIFeedbackExampleCode.m');
assert(exist(exampleFile,'file')==2, 'Example file not found: %s', exampleFile);

if ~exist(cfg.OutputDir,'dir'); mkdir(cfg.OutputDir); end

% Build scenario grid
[tb, bw, alt, seed] = ndgrid(cfg.TargetBler, cfg.BlerWindow, cfg.Altitude, cfg.Seeds);
N = numel(tb);
scenarios = repmat(struct('targetBler',[], 'blerWindow',[], 'altitude',[], 'seed',[]), N, 1);
for i=1:N
    scenarios(i).targetBler = tb(i);
    scenarios(i).blerWindow = bw(i);
    scenarios(i).altitude   = alt(i);
    scenarios(i).seed       = seed(i);
end

fprintf('[run_link_adaptation_sweeps] %d scenarios, SNR sweep %s, Frames=%d\n', ...
    numel(scenarios), vec2str(cfg.SNRRange), cfg.Frames);

% Choose execution mode
if cfg.UseParallel
    % Start pool if needed (choose pool type)
    pool = gcp('nocreate');
    if isempty(pool) || ~strcmpi(pool.Cluster.Type, cfg.PoolType)
        % Shut down existing pool if type differs
        if ~isempty(pool)
            delete(pool);
        end
        if strcmpi(cfg.PoolType,'Threads')
            parpool('Threads');
        else
            if isempty(cfg.Workers)
                parpool('Processes');
            else
                parpool('Processes', cfg.Workers);
            end
        end
    end

    % Asynchronous dispatch using parfeval (avoids parfor transparency limits)
    t0 = tic; total = numel(scenarios);
    futs(total,1) = parallel.FevalFuture;
    for i = 1:total
        futs(i) = parfeval(@doScenario, 4, exampleFile, cfg, scenarios(i));
    end

    done = 0; fails = 0;
    while done < total
        [~, ok, label, csvList, errmsg] = fetchNext(futs);
        done = done + 1;
        if ok
            fprintf('[%3d/%3d] %s OK  (%.1fs) CSVs: %s\n', done, total, label, toc(t0), strjoin(csvList, ', '));
        else
            fails = fails + 1;
            fprintf('[%3d/%3d] %s FAIL (%.1fs) %s\n', done, total, label, toc(t0), errmsg);
        end
    end
    fprintf('[run_link_adaptation_sweeps] Completed %d scenarios in %.1f min (failures=%d).\n', total, toc(t0)/60, fails);
else
    % Serial execution (no pool) — fastest startup for quick tests
    t0 = tic; total = numel(scenarios); fails = 0;
    for i = 1:total
        [ok, label, csvList, errmsg] = doScenario(exampleFile, cfg, scenarios(i));
        if ok
            fprintf('[%3d/%3d] %s OK  (%.1fs) CSVs: %s\n', i, total, label, toc(t0), strjoin(csvList, ', '));
        else
            fails = fails + 1;
            fprintf('[%3d/%3d] %s FAIL (%.1fs) %s\n', i, total, label, toc(t0), errmsg);
        end
    end
    fprintf('[run_link_adaptation_sweeps] Completed %d scenarios in %.1f min (failures=%d).\n', total, toc(t0)/60, fails);
end

end

%% Local helpers ----------------------------------------------------------
% No writeTempScript needed after we added overrides to the example.

function s = vec2str(v)
    % Render numeric vector compactly for logs and MATLAB literal
    if isscalar(v); s = num2str(v); return; end
    dif = diff(v);
    if all(abs(dif - dif(1)) < 1e-9)
        s = sprintf('%g:%g:%g', v(1), dif(1), v(end));
    else
        s = strjoin(arrayfun(@(x)sprintf('%g',x), v, 'UniformOutput',false), ' ');
    end
end

function tf = isCharLike(x)
    tf = ischar(x) || (isstring(x) && isscalar(x));
end

function [ok, label, csvList, errmsg] = doScenario(exampleFile, cfg, sc)
    label   = sprintf('TB%g_W%d_ALT%dkm_S%03d', sc.targetBler, sc.blerWindow, round(sc.altitude/1000), sc.seed);
    errmsg  = '';
    ok      = true;
    csvList = {};
    try
        outDir = fullfile(cfg.OutputDir, label);
        if ~exist(outDir,'dir'); mkdir(outDir); end
        % Prepare overrides for the example script
        NTN_OVERRIDE = struct(); %#ok<NASGU>
        NTN_OVERRIDE.NFrames = cfg.Frames;
        NTN_OVERRIDE.SNRIn = cfg.SNRRange;
        NTN_OVERRIDE.BLER_window = sc.blerWindow;
        NTN_OVERRIDE.TargetBLER = sc.targetBler;
        NTN_OVERRIDE.SatelliteAltitude = sc.altitude;
        NTN_OVERRIDE.OutputDir = outDir;
        NTN_OVERRIDE.Seed = sc.seed;
        NTN_OVERRIDE.LogEveryNSlots = cfg.LogEveryNSlots;
        run(exampleFile);
        csvs = dir(fullfile(outDir,'*.csv'));
        csvList = {csvs.name};
    catch ME
        ok = false;
        errmsg = getReport(ME,'basic','hyperlinks','off');
    end
end

function ap = makeAbsolutePath(p)
    if isempty(p)
        ap = pwd;
        return;
    end
    try
        jfile = java.io.File(p);
        if ~jfile.isAbsolute()
            p = fullfile(pwd, p);
            jfile = java.io.File(p);
        end
        ap = char(jfile.getCanonicalPath());
    catch
        % Fallback: simple heuristic by prefixing with pwd if not absolute
        if ispc
            isAbs = (~isempty(p) && ((numel(p) > 1 && p(2)==':') || startsWith(p,'\\')));
        else
            isAbs = startsWith(p, filesep);
        end
        if ~isAbs
            ap = fullfile(pwd, p);
        else
            ap = p;
        end
    end
end
