pathdep = 'G:\IP Project\IP Data Set\Test134Depth\';

for i=1:134
    i
    ipfile = strcat(pathdep,sprintf('test_Depthmap%d.mat',i));
    load(ipfile);
    testdm = Position3DGrid;
    testdm = testdm(:,:,4);
    testdm_ = imresize(testdm,[460,345]);

    dest = 'G:\IP Project\Project\testDepthmap\';
    opfile = strcat(dest,sprintf('test_depthmap%d',i));
    save(opfile,'testdm_');
end
