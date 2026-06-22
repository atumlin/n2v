function generate_references()
% generate_references - dump MATLAB GNNV (NNV) reach references for the
% cross-tool experiment.  Loops bus sizes x GNN types and calls the existing
% generate_ieee24_reference generator with matching instances / eps.
%
% Output: examples/GNN/experiments/results/refs/<type>_pf_<bus>_reference.mat
%
% NNV must be on the path.  Run:
%   matlab -batch "addpath(genpath('<nnv>')); addpath('<matlab_reference>'); ...
%                  cd('<experiments>'); generate_references()"

    here = fileparts(mfilename('fullpath'));
    refs_dir = fullfile(here, 'results', 'refs');
    if ~exist(refs_dir, 'dir'); mkdir(refs_dir); end

    ckpt_root = '/home/verivital/Anne/graph_verification/gnnv2/gnn_training/outputs';
    buses = {'ieee24', 'ieee39'};          % full-graph getRanges tractable here
    types = {'gcn', 'sage', 'gine_pretrain'};
    task = 'pf';
    instances = [1 2 3];                    % 1-based (MATLAB)
    eps_values = [1e-3 1e-2];

    for bi = 1:numel(buses)
        bus = buses{bi};
        for ti = 1:numel(types)
            typ = types{ti};
            ckpt = fullfile(ckpt_root, [bus '_' task], ...
                            [typ '_' task '_' bus '.mat']);
            out = fullfile(refs_dir, [typ '_' task '_' bus '_reference.mat']);
            if ~isfile(ckpt)
                fprintf('SKIP missing checkpoint: %s\n', ckpt); continue;
            end
            fprintf('\n==== %s / %s ====\n', bus, typ);
            try
                generate_ieee24_reference('checkpoint', ckpt, 'output', out, ...
                    'instances', instances, 'eps_values', eps_values);
            catch ME
                fprintf('ERROR on %s/%s: %s\n', bus, typ, ME.message);
            end
        end
    end
    fprintf('\nAll references written to %s\n', refs_dir);
end
