% Relies on the https://github.com/DIPlib/diplib

close all

A = readim('data/stained-glass-dark.jpg');
B = readim('data/stained-glass-light.jpg');

for l = 1:10
F = wfusimg(im2mat(A), im2mat(B), 'haar', l, 'mean', 'max');
imwrite(uint8(F), strcat('results/stained_glass_wfusimg_level', int2str(l), '.jpg'))
end

A = readim('data/brains/010/1.jpg');
B = readim('data/brains/010/1.jpg');
F = wfusimg(im2mat(A), im2mat(B), 'haar', 2, 'mean', 'max');
imwrite(uint8(F), strcat('results/brains_010_wfusimg.jpg'))

A = readim('data/brains/017/1.jpg');
B = readim('data/brains/017/1.jpg');
F = wfusimg(im2mat(A), im2mat(B), 'haar', 2, 'mean', 'max');
imwrite(uint8(F), strcat('results/brains_017_wfusimg.jpg'))

A = readim('data/brains/023/1.jpg');
B = readim('data/brains/023/1.jpg');
F = wfusimg(im2mat(A), im2mat(B), 'haar', 2, 'mean', 'max');
imwrite(uint8(F), strcat('results/brains_023_wfusimg.jpg'))

A = readim('data/brains/029/1.jpg');
B = readim('data/brains/029/1.jpg');
F = wfusimg(im2mat(A), im2mat(B), 'haar', 2, 'mean', 'max');
imwrite(uint8(F), strcat('results/brains_029_wfusimg.jpg'))

A = readim('data/brains/033/1.jpg');
B = readim('data/brains/033/1.jpg');
F = wfusimg(im2mat(A), im2mat(B), 'haar', 2, 'mean', 'max');
imwrite(uint8(F), strcat('results/brains_033_wfusimg.jpg'))

A = readim('data/clock1.jpg');
B = readim('data/clock2.jpg');
F = wfusimg(im2mat(A), im2mat(B), 'haar', 2, 'mean', 'max');
imwrite(uint8(F), strcat('results/clock_wfusimg.jpg'))
