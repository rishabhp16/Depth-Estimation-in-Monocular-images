function averagedepth= run_jnb(filename)
%clear all;close all;
addpath('utils');
addpath('bilateral');
addpath('UGM');
addpath('ksvdbox13');

load dict128Blur.mat;

%% Load data, set isProgagation = true if the original feature map contains obvious holes, otherwise false.
im = imread(filename);
isPropagation =true;

%figure,imshow(im, []);
if (ndims(im) == 3)
    im = rgb2gray(im);
end
im = double(im);

%% Compute original feature map
params.dict = D;
params.x = im;
params.maxatoms = 64;
params.sigma = 2;

rawMap = ompdenoise2(params,5);
params.maxatoms = max(rawMap(:));
%figure,imshow(rawMap, [1, params.maxatoms])

H = fspecial('gaussian', [5, 5], 1);
newMap = im(4:end-4,4:end-4)/255;

%% Feature Propagation
if (isPropagation)
    idx = (rawMap > 3);
    se = strel('disk',2);
    idx = imerode(idx,se);
    sigma_s = 15;
    sigma_r = 0.15;
    D = RF(rawMap.*idx, sigma_s, sigma_r, 3, imfilter(newMap, H));
    F = RF(double(idx), sigma_s, sigma_r, 3, imfilter(newMap, H));
    finalMap = bilateralC(D./F, imfilter(newMap, H), 5, 0.1);
else
    finalMap = RF(rawMap, 10, 0.2, 3, imfilter(newMap, H));
end

%% Final Result
%figure,imshow(finalMap, []);
averagedepth = blockproc(finalMap,[16 16],@(x)mean2(x.data));
%disp(averagedepth);
%disp(mean2(finalMap));
end

