
I = imread("C:/Users/liaml/Desktop/imageproc/demosaicing/images/mosaiced/truck.png");
GT = imread("C:/Users/liaml/Desktop/imageproc/demosaicing/images/ground_truth/truck.png");
I=I(:,:,1);
J = demosaic(I,"rggb");
imwrite(J, 'C:/Users/liaml/Desktop/imageproc/demosaicing/images/results/demosaiced_truck-matlab.png');

diff = GT - J;
squared_diff = diff .^ 2;

% Compute the mean squared error
mse = mean(squared_diff(:))

imshow(J)