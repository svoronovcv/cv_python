function [FASTFeat2, validFAST2, SURFFeat2, validSURF2, BRISKFeat2, validBRISK2] = extract_features(frame2g)
%Extract different features
SURFPts2 = detectSURFFeatures(frame2g);
FASTPts2 = detectFASTFeatures(frame2g);
BRISKPts2 = detectBRISKFeatures(frame2g);
[FASTFeat2, validFAST2] = extractFeatures(frame2g, FASTPts2);
[SURFFeat2, validSURF2] = extractFeatures(frame2g, SURFPts2);
[BRISKFeat2, validBRISK2] = extractFeatures(frame2g, BRISKPts2);
end