clear;clc;
file_path='..\datasets\ImageNet\';
N1=[];
imgDir  = dir([file_path '*.JPEG']);
for j=1:length(imgDir) 
    path=[file_path,imgDir(j).name];
    I=imread(path);
 if 3 == size(I,3)
    I1 = rgb2gray(I); % Transform RGB image to a Gray one.
  [rows,cols]=size(I1);

[A]=corner_detection(I1);
t=size(A,1);
N=[];
for i=1:30
    if t<100
      break;
    end
    if (A(i,2)+25<rows && A(i,2)-25>0 )&& (A(i,3)+25<cols &&A(i,3)-25>0)
     M=I(A(i,2)-25:A(i,2)+25,A(i,3)-25:A(i,3)+25,:);
     N(i,:,:,:)=M;
    end
end
 else
     N=[];
 end

N1=[N1;N];
disp(['now the image is' num2str(j) '.jpeg'])
end
save('./RCFIBS_datasets/RCFIBS.mat','N1')