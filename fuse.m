% Relies on the https://github.com/DIPlib/diplib

close all

sizes = [128, 256, 512, 724, 1024, 1448, 2048];
for l = 1:7
    A = rand(sizes(l), sizes(l), 3);
    B = rand(sizes(l), sizes(l), 3);
    times = zeros(100,1);
    for i = 1:100
        tic
        wfusimg(A, B, 'haar', l, 'mean', 'max');
        times(i) = toc;
    end
    strcat(int2str(sizes(l)),",wfusimg,", int2str(mean(times,'all')*1000),",", int2str(std(times,0,'all')*1000))
end
return

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
