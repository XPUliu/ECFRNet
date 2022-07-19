clear all
close all
clc

m1=load('./data/corner_position/test/lab/lab.mat');

I1=imread('./data/datasets/test/lab/lab.png');
n=ceil(m1.feature);

figure;
imshow(I1);

 hold on

 for j=1:249
  
     plot(round(n(j,1)),round(n(j,2)),'s','MarkerSize',6,'MarkerEdgeColor','c')
end

