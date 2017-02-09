path_dm = 'G:\IP Project\Project\trainDepthmap\';
target_depth = [];
for i=1:400
    i
    file = sprintf('train_depthmap%d.mat',i);
    load(strcat(path_dm,file));

    % create pacthes from loaded depth_map
    % patch = mat2cell(B,[16*ones(1,28),12],[16*ones(1,21),9]);
    
    %test_av_blur= reshape(test_av_blur',[1 numel(test_av_blur)]);
    patch = blockproc(B,[16 , 16],@(x)median(reshape(x.data,1,[])));
    patch = reshape(patch',1,numel(patch));
    % concatenate all patches 1-by-1 for current image
    
    target_depth = [target_depth ; patch];
end
save('target_depth.mat','target_depth');