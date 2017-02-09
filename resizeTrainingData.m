% Resizing training data to new folder trainResize with 460 X 345 resolution
for k = 1:134
    filename = sprintf('C:\\Users\\mudit\\Desktop\\Image Processing\\test134\\test%d.jpg',k);
    img = imread(filename);
    img = imresize(img,[460 345]);
    imwrite(img, sprintf('C:\\Users\\mudit\\Desktop\\Image Processing\\testResize\\test%d.jpg',k));
end
