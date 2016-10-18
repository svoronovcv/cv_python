clc
clear all
close all

I = imread('1.png');
Ihsv = double(I);

Imask = Ihsv(:,:,1)*0;
Ipart = Ihsv(:, :,:);
Ipartm = mean(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
Ipartstd = std(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
alpha = 1;
Inew = ((Ihsv(:,:,1) - Ipartm(1)).^2 < alpha*Ipartstd(1)^2) & ((Ihsv(:,:,2) - Ipartm(2)).^2 < alpha*Ipartstd(2)^2) &((Ihsv(:,:,3) - Ipartm(3).^2) < alpha*Ipartstd(3)^2);
imshow((1-Inew).*imgradient(I(:,:,3)) > 20)