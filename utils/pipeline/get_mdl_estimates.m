function out = get_mdl_estimates(mdl)
% out = get_mdl_estimates(mdl)

% default
if isempty(mdl)
    tmpmdl = fitglm(rand(10,2),rand(10,1));
    
    % nan out stuff
    B = nan;
    SE = nan;
    T = nan;
    Bp = nan;
    P = nan;
    S = nan;
    
    tmp = tmpmdl.devianceTest;
    v = tmp.Properties.VariableNames;
    
    mdl = [];
    ff = fields(tmpmdl);
    for ii=1:numel(ff)
        mdl.(ff{ii}) = tmpmdl.(ff{ii});
        if isstruct(tmpmdl.(ff{ii}))
            kk = fields(mdl.(ff{ii}));
            try
                for jj=1:numel(kk)
                    mdl.(ff{ii}).(kk{jj}) = nan(size(mdl.(ff{ii}).(kk{jj})));
                end
            end
        else
            mdl.(ff{ii}) = nan(size(mdl.(ff{ii})));
        end
    end
else
    
    % regressor specific
    B = mdl.Coefficients.Estimate;
    SE = mdl.Coefficients.SE;
    T = mdl.Coefficients.tStat;

    Bp = [];
    for ip=1:numel(B)
        h = zeros(1,numel(B));
        h(ip) = 1;
        Bp(ip) = coefTest(mdl,h);
    end

    % model specific
    tmp = mdl.devianceTest;
    v = tmp.Properties.VariableNames;
    P = tmp.pValue(2);
    S = tmp.(v{3})(2);
end

% output
out = [];
out.B = B;
out.SE = SE;
out.T = T;
out.Bp = Bp;
out.mdl_p = P;
out.(['mdl_' v{3}]) = S;
out.CoefficientNames = mdl.CoefficientNames;
out.Rsquared = mdl.Rsquared;
out.LogLikelihood = mdl.LogLikelihood;
out.Deviance = mdl.Deviance;
out.Dispersion = mdl.Dispersion;
out.SSR = mdl.SSR;
out.SST = mdl.SST;
out.SSE = mdl.SSE;
out.DFE = mdl.DFE;
out.CoefficientCovariance = mdl.CoefficientCovariance;
out.ModelCriterion = mdl.ModelCriterion;


