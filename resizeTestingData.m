% Resizing testing data to new folder testResize with 460 X 345 resolution

for k = 1:400
    filename = sprintf('C:\\Users\\mudit\\Desktop\\Image Processing\\test134\\test%d.jpg',k);
    img = imread(filename);
    img = imresize(img,[460 345]);
    imwrite(img, sprintf('C:\\Users\\mudit\\Desktop\\Image Processing\\testResize\\test%d.jpg',k));
end
