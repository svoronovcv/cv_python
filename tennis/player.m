clc
clear all
close all

I = imread('f.jpg');
Ihsv = rgb2hsv(I);

Imask = Ihsv(:,:,1)*0;
Ipart = Ihsv(:, :,:);
Ipartm = mean(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
Ipartstd = std(reshape(Ipart,[size(Ipart,1)*size(Ipart,2) 3]));
alpha = 6;
Inew = (abs(Ihsv(:,:,1) - Ipartm(1)) < alpha*Ipartstd(1)^2) &...
    (abs(Ihsv(:,:,2) - Ipartm(2)) < alpha*Ipartstd(2)^2);
imshow((1-Inew))