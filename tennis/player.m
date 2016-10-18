clc
clear all
close all

I = imread('n.jpg');
Ihsv = rgb2hsv(I);

Imask = Ihsv(:,:,1)*0;
Ipart = Ihsv(550:900, 500:1280,:);
Ipartm = mean(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
Ipartstd = std(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
alpha = 0.8;
Inew = ((Ihsv(:,:,1) - Ipartm(1)).^2 < alpha*Ipartstd(1)) & ((Ihsv(:,:,3) - Ipartm(3)).^2 < alpha*Ipartstd(3));
imshow(1-Inew)