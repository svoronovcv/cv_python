function [C] = interpcoord(limits, data)
C = data;
ob = (data > limits(2)) | (data < limits(1));
for i =2:length(data)-1
    if(ob(i) > 0)
        flag = true;
        len = 1;
        while flag && (i + len < length(data))
            if(ob(i+len) < 1)
                flag = false;
            else
                len = len + 1;
            end
        end
        del = 1/(len+1);
        C(i) = C(i-1)*(1 - del) + data(i+len)*del;
    end
    ob(i) = 0;
end
end

