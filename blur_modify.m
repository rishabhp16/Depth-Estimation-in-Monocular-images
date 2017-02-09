blur_input_ann = zeros(400,638);
load('avblur_train.mat');
blur_train = permute(blur_train,[3 1 2]);
for i=1:400
    i
    temp = permute(blur_train(i,:,:),[1 3 2]);
    blur_input_ann(i,:) = reshape(temp,[1 numel(blur_train(i,:,:))]);
end
save('blur_input_ann','blur_input_ann');