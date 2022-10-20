function [S,Sr]=get_stability(Qlab,Qrlab,lags,idat,nrand,nstate)
% [S,Sr]=get_stability(Qlab,Qrlab,lags,idat,nrand,nstate)

S = nan(numel(lags)-1,max(idat));
Sr = nan(numel(lags)-1,max(idat),nrand);

irep = 0;
nsmp = numel(S);
fprintf('observed stability')
for id=1:max(idat)
    for il1=1:numel(lags)-1
        irep=irep+1;
        dotdotdot(irep,0.1,nsmp)
        il2 = il1+1;

        %q1 = squeeze(Qlab(il,id,:));
        q1 = squeeze(Qlab(il1,id,:))+1;
        q2 = squeeze(Qlab(il2,id,:))+1;

        bad = isnan(q1);
        mx = ami(q1(~bad),q2(~bad));

        % store
        S(il1,id) = mx;
        foo=1;

       % random
       if 0
        good = find(~bad);
        qr2 = nan(size(q2));
        for ir=1:nrand
            idx = good( randperm(numel(good)) );
            qr2(good) = q2(idx);
            mx = ami(q1(~bad),qr2(~bad));
            Sr(il1,id,ir) = mx;
        end
       end
    end
end
fprintf('\n')

% random
if 1
    fprintf('rand stability')
    Sr = nan(numel(lags)-1,max(idat),nrand);
    nsmp = numel(Sr);
    irep = 0;
    for ir=1:nrand
        for id=1:max(idat)
            for il1=1:numel(lags)-1
                irep=irep+1;
                dotdotdot(irep,0.1,nsmp)
                il2 = il1+1;

                q1 = squeeze(Qrlab(il1,id,ir,:))+1;
                q2 = squeeze(Qrlab(il2,id,ir,:))+1;

                bad = isnan(q1);
                mx = ami(q1(~bad),q2(~bad));


                % store
                Sr(il1,id,ir) = mx;
                foo=1;
            end
        end
    end
end