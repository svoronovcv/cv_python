clc
clear all
close all
% Load data
gaze = load('/home/pc/Desktop/MainExp/gaze.mat');
startingTime = load('/home/pc/Desktop/MainExp/gazeTimestamp.mat');
workingDir = dir('/home/pc/Desktop/MainExp/');
DirNames = {workingDir.name}';
Names = {'Ajung', 'Camille', 'Denys', 'Iason', 'Jose', 'Laura', 'Mahdi', 'Murali', 'Nili', 'Sylvain'};
% S = struct();
% S.Time = cell(size(gaze.Time));
% S.BPORX = cell(size(gaze.BPORX));
% S.BPORY = cell(size(gaze.BPORY));
S = load('newgaze_Snfull.mat');
figure
for j = 3:3 % length(DirNames) % iterations for names
    k = findNameidx(DirNames(j), Names);
    if(isdir(char(strcat('/home/pc/Desktop/MainExp/', DirNames(j)))))
        trial = findTrialnum(DirNames(j));
        Dir = char(strcat('/home/pc/Desktop/MainExp/', DirNames(j)));
        imageNames = dir(strcat(Dir,'/*.jpg'));
        imageNames = {imageNames.name}';
        k, trial
        if(k > 0 && length(gaze.Time{1, k}{1, trial}) > 0 && length(imageNames) > 3000)
            tm = gaze.Time{1, k}{1, trial};
            initStart = startingTime.gazeStartTimestamp{1, k}(trial);
            Xcoord = gaze.BPORX{1, k}{1, trial};
            Ycoord = gaze.BPORY{1, k}{1, trial};
            Xnew = interpcoord([0 1280], Xcoord);
            Ynew = interpcoord([0 960], Ycoord);
            x = 1:30/24:length(Xnew);
            y = 1:30/24:length(Ynew);
            pts = 1:length(Xnew);
            Xi = interp1q(pts',Xnew,x');
            Yi = interp1q(pts',Ynew,y');
            newX = Xi;
            newY = Yi;
            firstFrame = findStart(initStart, tm, imageNames)
            for t = 1:1 % iterations for trials
                a = firstFrame+163*t; % start and end of a trial
                b = a + 80; %100;
                frame1 = imread(fullfile(Dir,imageNames{a}));
                frame1g = rgb2gray(frame1);
                %             FASTPts1 = detectSURFFeatures(frame1g);
                %             [FATFeat1, validFAST1] = extractFeatures(frame1g, FASTPts1);
                [FASTFeat1, validFAST1, SURFFeat1, validSURF1, BRISKFeat1, validBRISK1] = extract_features(frame1g);
                % imshow(frame1)
                % hold on
                % plot(829,603,'r.','MarkerSize',10)
                % hold off
                H = eye(3);
                U = H;
                dist = zeros(length(imageNames),1);
                oldcoords = [];
                itka = 0;
                Figure = figure;
                for i = a+1:b%length(imageNames) % iterations for images in the trial
                    itka = itka+1;
                    oldcoord = [Xi(i + length(Xi)-length(imageNames)) Yi(i + length(Xi)-length(imageNames))];
                    frame2 = imread(fullfile(Dir,imageNames{i}));
                    frame2g = rgb2gray(frame2);
                    [transform, inlierPtsDistorted,inlierPtsOriginal, status, FASTFeat2, ...
                        validFAST2, SURFFeat2, validSURF2, BRISKFeat2, validBRISK2] = ...
                        featmatch3(frame2g, FASTFeat1, validFAST1, SURFFeat1, validSURF1,BRISKFeat1, validBRISK1); % match the features
                    FASTFeat1 = FASTFeat2; % reassign the features
                    validFAST1 = validFAST2;
                    SURFFeat1 = SURFFeat2;
                    validSURF1 = validSURF2;
                    BRISKFeat1 = BRISKFeat2;
                    validBRISK1 = validBRISK2;
                    
                    %                 trnsl(i, :) = [transform.T(3,1) transform.T(3,2)];
%                     W = transformPointsForward(projective2d((transform.T*H)^-1), oldcoord);
%                     dist(i) = abs(oldcoord(1) - W(1));
                    if(status > 0)
                        transform.T = U;
                    end
                    if(rcond(transform.T*H) < 10^-5)
                        %         H = eye(3);
                        %         newcoord = oldcoord;
                           transform.T = eye(3);                   
%                     elseif(dist(i) > 400)
%                         H = eye(3);
                    else
                        H = transform.T*H;
                    end
                    %     oldcoord = [829 603];
					% compute the new coordinates
                    U = transform.T;
                    newcoo = transformPointsForward(projective2d(H), oldcoord);
                    newcoord = transformPointsForward(projective2d(H), oldcoord);
                    newX((i + length(Xi)-length(imageNames))) = newcoord(1);
                    newY((i + length(Xi)-length(imageNames))) = newcoord(2);
                    outputView = imref2d(size(frame2)); % display the transformed image
                    Fr = imwarp(frame2, projective2d(H),'OutputView', outputView);
                        imshow(frame2)
                        hold on
%                         plot(oldcoord(1), oldcoord(2), 'bo','MarkerSize',10, 'LineWidth',10)
%                     viscircles([oldcoord(1), oldcoord(2)], [5])
%            oldcoords = [Xi(a+1 + length(Xi)-length(imageNames):i+length(Xi)-length(imageNames)) Yi(a+1 + length(Xi)-length(imageNames):i+length(Xi)-length(imageNames))];
                    c = [oldcoord(1)-50, oldcoord(2)-50];
                    oldcoords = [oldcoords ; c]; %transformPointsForward(projective2d(H), c)];
                    A = 7;
                    B = 36;
                    C = 39;
%                     if (itka <A)            
                        plot(oldcoords(:,1), oldcoords(:,2), 'y','LineWidth',6)
%                     elseif itka < B
%                         plot(oldcoords(1:A,1), oldcoords(1:A,2), 'y','LineWidth',6)
%                         plot(oldcoords(A:end,1), oldcoords(A:end,2), 'r','LineWidth',6)
%                     elseif itka <C
%                         plot(oldcoords(1:A,1), oldcoords(1:A,2), 'y','LineWidth',6)
%                         plot(oldcoords(A:B,1), oldcoords(A:B,2), 'r','LineWidth',6)
%                         plot(oldcoords(B:end,1), oldcoords(B:end,2), 'y','LineWidth',6)
%                     else
%                         plot(oldcoords(1:A,1), oldcoords(1:A,2), 'y','LineWidth',6)
%                         plot(oldcoords(A:B,1), oldcoords(A:B,2), 'r','LineWidth',6)
%                         plot(oldcoords(B:C,1), oldcoords(B:C,2), 'y','LineWidth',6)
%                         plot(oldcoords(C:end,1), oldcoords(C:end,2), 'r','LineWidth',6)
%                     end

                    
                    r = 8;
                    pos = [c-r 2*r 2*r];
%                     r = rectangle('Position',pos,'Curvature',[1 1], 'FaceColor', 'b', 'Edgecolor','none');
% %                         plot(newcoo(1), newcoo(2) ,'b*','MarkerSize',20)
%                         pause
                        saveas(Figure,char(strcat('/media/pc/ntfs/for_pres/',string(itka),'-raw.jpg')))
%                         imwrite(Fr, char(strcat('/media/pc/ntfs/for_pres/',string(itka),'.jpg')),'Quality',100);
                        hold off
                end
            end
%             xk = 1:24/30:length(newX);
%             pts = 1:length(newX);
%             XXi = interp1q(pts',newX,xk')';
%             YYi = interp1q(pts',newY,xk')';
%             S.Time{1, k} {1, trial} = gaze.Time{1, k} {1, trial};
%             S.BPORX{1, k} {1, trial} = XXi;
%             S.BPORY{1, k} {1, trial} = YYi;
        end
    end
end
% save('newgaze_Snfull-new.mat','-struct','S');