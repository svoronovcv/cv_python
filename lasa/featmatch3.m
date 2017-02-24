function [transform, inlierPtsDistorted,inlierPtsOriginal, status, FASTFeat2, validFAST2, SURFFeat2, validSURF2, BRISKFeat2, validBRISK2] = featmatch3(frame2g, FASTFeat1, validFAST1, SURFFeat1, validSURF1,BRISKFeat1, validBRISK1)
%Perform feature matching
SURFPts2 = detectSURFFeatures(frame2g);
FASTPts2 = detectFASTFeatures(frame2g);
BRISKPts2 = detectBRISKFeatures(frame2g);
[FASTFeat2, validFAST2] = extractFeatures(frame2g, FASTPts2);
[SURFFeat2, validSURF2] = extractFeatures(frame2g, SURFPts2);
[BRISKFeat2, validBRISK2] = extractFeatures(frame2g, BRISKPts2);
matchedFAST = matchFeatures(FASTFeat1, FASTFeat2, 'Unique', true);
matchedOFAST = validFAST1(matchedFAST(:, 1), :);
matchedDFAST = validFAST2(matchedFAST(:, 2), :);
matchedSURF = matchFeatures(SURFFeat1, SURFFeat2, 'Unique', true);
matchedOSURF = validSURF1(matchedSURF(:, 1), :);
matchedDSURF = validSURF2(matchedSURF(:, 2), :);
matchedBRISK = matchFeatures(BRISKFeat1, BRISKFeat2, 'Unique', true);
matchedOBRISK = validBRISK1(matchedBRISK(:, 1), :);
matchedDBRISK = validBRISK2(matchedBRISK(:, 2), :);
matchedO  = [matchedOFAST.Location; matchedOSURF.Location; matchedOBRISK.Location];
matchedD = [matchedDFAST.Location; matchedDSURF.Location; matchedDBRISK.Location];
[transform, inlierPtsDistorted,inlierPtsOriginal, status] = estimateGeometricTransform(matchedD, matchedO, 'projective', 'MaxNumTrials', 1000, 'Confidence', 99);
end

