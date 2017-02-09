path = 'G:\IP Project\Project\trainResize\';
hue_data =[];
sat_data = [];
val_data =[];
for i=1:400
    i
    filename=strcat(path,sprintf('trainR%d.jpg',i));
    img=imread(filename);
    color_=rgb2hsv(img);
    hue=color_(:,:,1);
    sat=color_(:,:,2);
    val=color_(:,:,3);
    p = blockproc(hue,[16 16],@(x)mean2(x.data));
    hue_data = [hue_data; reshape((p)',1,numel(p)) ]; 
    p = blockproc(sat,[16 16],@(x)mean2(x.data));
    sat_data =  [sat_data ; reshape((p)',1,numel(p))];
    p = blockproc(val,[16 16],@(x)mean2(x.data));
    val_data = [val_data ; reshape((p)',1,numel(p)) ];
end
save('color_cues.mat','hue_data','sat_data','val_data');