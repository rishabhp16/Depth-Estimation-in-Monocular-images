path = 'G:\IP Project\Project\trainResize\';
filename=strcat(path,sprintf('trainR1.jpg'));
first=run_jnb(filename);
blur_train(:,:,1)=first;
size(blur_train)
for i = 2:400
    i
    filename=strcat(path,sprintf('trainR%d.jpg',i));
    img = imread(filename);
   % pause(1);
    %imshow(img);
    temp=run_jnb(filename);
    blur_train=cat(3,blur_train,temp);
end

save('avblur_train.mat','blur_train');