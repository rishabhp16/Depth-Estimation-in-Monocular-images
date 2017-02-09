pathdep = 'G:\IP Project\IP Data Set\Train400Depth\';

for i=1:3
    i
    ipfile = strcat(pathdep,sprintf('train_depthmap%d.mat',i));
    load(ipfile);
    traindm = Position3DGrid;
    traindm = traindm(:,:,4);
    traindm_ = imresize(traindm,[460,345]);

    dest = 'G:\IP Project\Project\trainDepthmap\';
    opfile = strcat(dest,sprintf('train_depthmap%d',i));
    %save(opfile,'traindm_');
end
