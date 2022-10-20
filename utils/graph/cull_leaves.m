function [T_new,newval,oldval] = cull_leaves(T,mx)
% [T_new,newval,oldval] = cull_leaves(T,mx)

% get bad leaves
bad = isnan(T(:,3));

tmpz = T(~bad,1:2);
tmpz=tmpz(:);
selgood = ismember(1:mx,tmpz(tmpz<=mx));
mx_new = sum(selgood);

% remap values
oldval = 1:mx;
newval = nan(size(oldval));
newval(selgood) = 1:mx_new;

T_new = T(~bad,1:2);
T_new = changem(T_new,newval,oldval);
T_new(T_new>mx_new) = T_new(T_new>mx_new)-mx+mx_new;
T_new = [T_new, T(~bad,3)];
         