path = 'G:\IP Project\Project\trainResize\';
% GIST Parameters:
clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Pre-allocate gist:
Nfeatures = sum(param.orientationsPerScale)*param.numberBlocks^2;
gist_val= zeros([400 Nfeatures]); 

% Load first image and compute gist:
filename=strcat(path,'trainR1.jpg');
img = imread(filename);
gist=struct('img','val');
[gist_val(1, :), param] = LMgist(img, '', param); % first call
gist(1).img=1;
gist(1).val=gist_val(1,:);
%disp(gist(1,:));
% Loop:
for i = 2:400
    i
    filename=strcat(path,sprintf('trainR%d.jpg',i));
    img = imread(filename);
   % pause(1);
    %imshow(img);
    gist_val(i, :) = LMgist(img, '', param); % the next calls will be faster
    gist(i).img=i;
    gist(i).val=gist_val(i,:);
end
save('gist_features.mat','gist');
sprintf('GIST features saved.')
