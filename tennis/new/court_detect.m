clc

I = imread('n1.jpg');
% Ihsv = rgb2hsv(I);
Iv = I(:,:,3);
Im1 = zeros(size(Iv));
Im1(180:end,260:380) = 1;
Ipart = (I(:,:,1) >150).*(I(:,:,2) >150).*(I(:,:,3) >100).*Im1;
Ibw = Ipart;
maxs=0;
for i=260:380
    for j = 200:350
        Ip = Ibw(j-7:j+7,i-5:i+4);
        s = sum(sum(Ip == mask));
        if(s >=maxs)
            maxs=s;
            mi=i;
            mj=j;
        end
    end
end
imshow(Ibw)
hold on
plot(mi,mj,'b*','MarkerSize',10)