clc
clear all
close all

I = imread('2.png');
Ihsv = rgb2hsv(I);

Imask = Ihsv(:,:,1)*0;
Ipart = Ihsv(500:end, :,:);
Ipartm = mean(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
Ipartstd = std(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
alpha = 15;
Inew = ((Ihsv(:,:,1) - Ipartm(1)).^2 < alpha*Ipartstd(1)^2) &...
    ((Ihsv(:,:,2) - Ipartm(2).^2) < alpha*Ipartstd(2)^2);
imshow((1-Inew))