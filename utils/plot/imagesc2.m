function varargout = imagesc2(varargin)
% wrapper for imagesc. excepts all the smae inputs but enforces proper
% spacing

if nargin==1
    h = imagesc(varargin{:});
else
    if ischar(varargin{1})
        h = imagesc(varargin{:});
    else % x,y,C + args, enforce spacing
        x = varargin{1};
        y = varargin{2};
        C = varargin{3};
        
        % spacing
        Cs = C;
        s = round(diff(x)/min(diff(x)));
        xs = x(1):min(diff(x)):x(end); % = 0:25:800
        Cs = repelem(Cs,1,s([1 1:end]),1);

        s = round(diff(y)/min(diff(y)));
        ys = y(1):min(diff(y)):y(end); % = 0:25:800
        Cs = repelem(Cs,s([1 1:end]),1,1);
        
        % plot
        args = varargin(4:end);
        h = imagesc(xs,ys,Cs,args{:});
        
        foo=1;
    end
end

%% output
if nargout>0
    varargout{1} = h;
end