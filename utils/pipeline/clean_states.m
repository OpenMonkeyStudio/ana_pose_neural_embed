function [Cout,stateMap] = clean_states(C,minSmp,minP)
% Cout = clean_states(C,minSmp,minP)
%
% interp states that are too rare or missing


% clean rare states
n = accumarray(C,1);
p = n ./ sum(n);

good = find(p>minP);

% remap
oldval = 1:max(C);
newval = nan(size(oldval));
newval(good) = 1:numel(good);
C = changem(C,newval,oldval);

stateMap = [oldval;newval];

% find state indices
istate = find(diff(C)~=0);

% find states that last only X amount
ds = diff(istate);
ibad = find(ds <= minSmp);

% interp over these segments
Cout = C;
for ib=1:numel(ibad)
    st = istate(ibad(ib))+1;
    fn = st+ds(ibad(ib))-1;
    
    Cout(st:fn) = nan;
end
imiss = find(isnan(Cout));
igood = find(~isnan(Cout));

Ci = interp1(igood,C(igood),imiss,'nearest','extrap');
Cout(imiss) = Ci;

