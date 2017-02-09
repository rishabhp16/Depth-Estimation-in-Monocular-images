path_dm_test = 'G:\IP Project\Project\testDepthmap\';
test_depth = [];
for i=1:134
    i
    file = sprintf('test_depthmap%d.mat',i);
    load(strcat(path_dm_test,file));
    % returns obect with name testdm_
    
    patch = blockproc(testdm_,[16 , 16],@(x)median(reshape(x.data,1,[])));
    patch = reshape(patch',1,numel(patch));
    % concatenate all patches 1-by-1 for current image
    
    test_depth = [test_depth ; patch];
end
save('test_depth.mat','test_depth');