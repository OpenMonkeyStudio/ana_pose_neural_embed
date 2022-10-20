function varargout = signtest_dim(dim,varargin)
% [P,H,STAT] = signtest_dim(dim,varargin)
%
% dim: diumension to loop over (eg if dim=1, signtest will be applied to
% rows, NOT columns)

% init
%NB: to work as the rest of matlab does, dim=1 should apply the test over
%individual columns... butthat means we have to index over the OTHER
%dimension
if isempty(dim)
    dim = 2; % loop over rows
else
    if dim==1; dim=2;
    elseif dim==2; dim=1;
    else, error('only 2D matrices!')
    end
end

X = varargin{1};
varargin(1) = [];

if numel(varargin)>1
    Y = varargin{2};

    if isdouble(y)
       varargin(1) = [];
    else
        Y = [];
    end
else
    Y = [];
end

P = [];
H = [];
S = [];

% loop over the indicated dimension
for ii=1:size(X,dim)
    idx = {':',':'};
    idx{dim} = ii;
    
    x = X(idx{:});
    if ~isempty(Y) && numel(Y) > 1
       y = Y(idx{:});
       [p,h,s] = signtest(x,y,varargin{:});
    else
        [p,h,s] = signtest(x,varargin{:});
    end
    P = cat(dim,P,p);
    H = cat(dim,H,h);
    S = cat(dim,S,s);
    
end

%output
varargout{1} = P;
if nargout>1; varargout{2} = H;end
if nargout>2; varargout{3} = S;end
