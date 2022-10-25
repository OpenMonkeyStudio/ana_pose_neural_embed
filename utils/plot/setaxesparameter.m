function setaxesparameter(varargin)
% setaxesparameter(parameter)
% setaxesparameter(h,parameter)
% setaxesparameter(h,parameter,newval)
% setaxesparameter(h,parameter1,newval1,parameter2,newval2,...)
%
% if h is empty, then plots into all current axes

%assign inputs
if all( ishandle(varargin{1}(:)) ) || isempty(varargin{1})
    h = varargin{1};
    varargin(1) = [];
else
    h = [];
end

while numel(varargin)>0
    parameter = varargin{1};
    varargin(1) = [];
    if numel(varargin)>0
        newval = varargin{1}; 
        varargin(1) = [];
    else
        newval = [];
    end
    
    if isempty(h); h = findobj(gcf,'type','axes'); end
    h = h(:);

    %get currentvalue
    vals1 = get(h,parameter);

    %set value
    if any( strcmpi(parameter,{'ylim','xlim','clim','zlim','rlim','thetalim'}) )
        %disp( ['calculating ' parameter] )
        %get new value
        if ~isempty(newval) && ~any(isnan(newval))
            vals2 = newval;
        else
            if numel(h) > 1
                mn = min( cellfun(@min,vals1) );
                mx = max( cellfun(@max,vals1) );

                vals2 = [mn, mx];
            else
                vals2 = vals1;
            end
        end
        
        % if nan in newval, replcae the nan
        if ~isempty(newval)
           good = ~isnan(newval);
           vals2(good) = newval(good);
        end
        
        %set new value
        %disp( ['New ' parameter ': ' mat2str(vals2)] )
        set(h,parameter,vals2)
    elseif strcmpi(parameter,'fontsize')
        set(h,parameter,newval)
    else
        error(['this parameter not coded for: ' parameter])
    end
end
        