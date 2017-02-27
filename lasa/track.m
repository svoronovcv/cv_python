close all
clear all

coords = load('Markers_coords.txt');
initPoints = [
    15.3 26.9;
    6.2 23;
    23.5 21.5;
    14 16.1;
    1.9 14.8;
    28 12;
    5.5 8.5;
    14 4.8;
    23 2.7;
    
    0 0;
    16.5 -3.5;
    29.5 -3.6;

    21.5 -9.6;
    5.5 -10.8;
    26.5 -9.6;28 2.7;33 12; 19 16.1
    ];
A = [
    234 119;
    207 128;
    265 125;
    238 139;
    195 146;
    290 143;
    211 160;
    245 167;
    284 168;
    191 188;
    266 191;
    321 185;
    298 210;
    222 225;
    321 207;
    304 167;
    307 142;
    254 137;
    ];
% initP = [21.5  -9.6; 26.5 -9.6;  23 2.7;    28 2.7];
% imgP =  [297    210; 321   207; 283 169;   303 167];
initP = [20.5  -8.8; 25.5 -8.8; 19 -22.1; 24 -22.1];
imgP =  [295    204; 318   200; 319 266;   348 261];
% for i = 1:72
%     X(:,i) = coords(:,i*2);
%     Y(:,i) = coords(:,i*2+1);
%     names = coords(:,1);
% end
for i = 72:72
    I = imread(strcat('/media/pc/ntfs/Motion correction experiment/Exp4/wearcam-2017-02-10_001/' ...
        , char(string(i)),'-matlab.jpg'));
    Tr = estimateGeometricTransform(initP, imgP,'projective', ...
        'MaxNumTrials', 2000,'Confidence', 99);
    newp = transformPointsForward(Tr, initPoints(:,:));
    imshow(I)
    hold on
%     plot(X(i,14:15),Y(i,14:15), 'wx', 'MarkerSize', 10,'LineWidth', 2),
    plot(newp(:,1),newp(:,2), 'rx','MarkerSize', 10,'LineWidth', 2)
    plot(A(:,1),A(:,2), 'yx','MarkerSize', 8,'LineWidth', 2)
    plot(imgP(:,1),imgP(:,2), 'bx','MarkerSize', 10,'LineWidth', 2)
%     avg_dist(i) = sum(sqrt((Xt(i,:)'-newp(:,1)).^2 + (Yt(i,:)'-newp(:,2)).^2))/14;
%     max_dist(i) = max(sqrt((Xt(i,:)'-newp(:,1)).^2 + (Yt(i,:)'-newp(:,2)).^2));
    pause(0.1)
    hold off
end
dist = (sqrt((A(:,1)-newp(:,1)).^2 + (A(:,2)-newp(:,2)).^2));
avg_dist = sum(sqrt((A(:,1)-newp(:,1)).^2 + (A(:,2)-newp(:,2)).^2))/14;
max_dist = max(sqrt((A(:,1)-newp(:,1)).^2 + (A(:,2)-newp(:,2)).^2));
% figure, plot(avg_dist), hold on, plot(max_dist)
% Xt = zeros(169,14);
% Yt = zeros(169,14);
% Xt(1,:) = X(1,:);
% Yt(1,:) = Y(1,:);
% Xt(:,1) = X(:,1);
% Yt(:,1) = Y(:,1);
% for i = 2:169
%     for j = 2:14
%         mind = 10000;
%         for t = 2:14
%             dist = (Xt(i-1,j) - X(i,t))^2 + (Yt(i-1,j) - Y(i,t))^2;
%             if( dist < mind)
%                 mind = dist;
%                 Xt(i,j) = X(i,t);
%                 Yt(i,j) = Y(i,t);
%             end
%         end
%     end
% end
% p1 = 10;
% p2 = 14;
% p3 = 4;
% p4 = 6;
% avg_dist = zeros(169,1);
% max_dist = zeros(169,1);
% for i = 1:169
%     I = imread(strcat('/media/pc/ntfs/Motion correction experiment/Exp3/wearcam-2017-02-01_005/' ...
%         , char(string(names(i))),'-matlab.jpg'));
%     Tr = estimateGeometricTransform([initPoints(p1,:); initPoints(p2,:); initPoints(p3,:); initPoints(p4,:)]...
%         , [Xt(i,p1)' Yt(i,p1)';Xt(i,p2)' Yt(i,p2)';Xt(i,p3)' Yt(i,p3)';Xt(i,p4)' Yt(i,p4)'],'projective', ...
%         'MaxNumTrials', 1000,'Confidence', 99);
%     newp = transformPointsForward(Tr, initPoints(:,:));
%     imshow(I)
%     hold on
%     plot(Xt(i,:),Yt(i,:), 'wx', 'MarkerSize', 10,'LineWidth', 2),
%     plot(newp(:,1),newp(:,2), 'rx','MarkerSize', 8,'LineWidth', 2)
%     avg_dist(i) = sum(sqrt((Xt(i,:)'-newp(:,1)).^2 + (Yt(i,:)'-newp(:,2)).^2))/14;
%     max_dist(i) = max(sqrt((Xt(i,:)'-newp(:,1)).^2 + (Yt(i,:)'-newp(:,2)).^2));
%     pause(0.1)
%     hold off
% end
% figure, plot(avg_dist), hold on, plot(max_dist)