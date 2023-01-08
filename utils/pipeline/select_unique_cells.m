function [isFirst,area,days] = select_unique_cells(monk)
% [isFirst,area,days] = select_unique_cells(monk)

% import cell info
[area,yield,area2label,days] = import_cells(monk);

% first recorded cells
isFirst = zeros(size(yield));
for ich=1:size(yield,1)
    tmp = yield(ich,:);
    
    dy = diff(tmp);
    da = diff(area(ich,:));
    flag = find(dy==1)+1;
    
    tmp2 = zeros(1,numel(tmp));
    tmp2(flag) = 1;
    isFirst(ich,:) = tmp2;
end

