start_all=tic;

if 0
    %test_cca_wnn_alignment5
    ana_embedding2
    ana_modularity5c
    %ana_symposium2
    %ana_modularity5b
else
    if 1
        %test_cca_wnn_alignment6
        ana_embedding3b
        ana_modularity5c
    end
    %get_all_regressors
    ana_neural_embed
    % ana_dist
    % ana_encode_lag2
end

fn=toc(start_all);
fprintf('WHOLE ANALYSIS TIME: %g\n',fn)