function  [locs]=corner_detection(im)
tic
I=im;

% if mod(c,2)==0 && mod(r,2)==0
%  r=r+1;c=c+1;
%   I3=zeros(r,c);
%    I3(2:r,2:c)=I;
%    I4=I3;
% elseif mod(c,2)==0
%     c=c+1;
%      I2=zeros(r,c);
%    I2(r,2:c)=I;
%    I4=I2;
% elseif mod(r,2)==0
%     r=r+1;
%     I1=zeros(r,c);
%    I1(2:r,c)=I; 
%    I4=I1;
% else
%     I4=I;
% end
 

  [measure,imge_map]= affine3(I);
  [rr,cc]=find(imge_map==1);
  N=[];
for i=1:length(rr)
        KKK=measure(rr(i),cc(i));
        N1=[KKK,rr(i),cc(i)];
        N=[N;N1];
end


if isempty(N)
    N2=[];
else
 N2=sortrows(N,-1);
end
 locs=N2;
 toc
end
function [measure,marked_img]=improved3(im)


p = 8;
rho1 = 1;
sigma1 =1.5;
  threshold = 1*10^9;



% im = rgb2gray(im);
[rows cols] = size(im);
wid = 15;
width=7;
%%%%%%%%% Extend the image %%%%%%%%%
extend_image = zeros(rows+2*wid,cols+2*wid);

candidate = zeros(rows,cols);
% SS =  zeros(rows+2*wid,cols+2*wid);

extend_image(wid+1:rows+wid,wid+1:cols+wid) = im;

for i = 1:size(extend_image,1)                         
    if i<=wid
        r = 2*wid+1-i;
    elseif i>rows + wid
        r = 2*(rows+wid)+1-i;
    else
        r = i;
    end
    for j = 1:size(extend_image,2)
        if j<=wid
            c = 2*wid+1-j;
        elseif j>cols + wid
            c = 2*(cols+wid)+1-j;
        else
            c = j;
        end
        extend_image(i,j) = extend_image(r,c);
    end
end
im=extend_image;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 
  anigs_direction(1:(2*width+1),1:(2*width+1),p) = 0;

for direction = 1:p
    theta = (direction-1)*pi/p;
    for x = -width:1:width
        for y = -width:1:width
            xr = x*cos(theta)+y*sin(theta);
            yr = -x*sin(theta)+y*cos(theta);
         
            anigs_direction(x+width+1,y+width+1,direction) = ...
          (1/(2*pi*sigma1))*exp((-1/(2*sigma1))*(xr^2*rho1 + yr^2/rho1))...
            *(rho1/sigma1)*((rho1/sigma1)*(xr^2)-1);
    
        end
    end
end
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%?smooth the input image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  template(1:2*wid+rows,1:cols+2*wid,p) = 0; 
  
%%% the kernal convolution with the original image


for direction=1:p
    oa = anigs_direction(:,:,direction);
    oa = oa - sum(oa(:))/numel(oa);
    template(:,:,direction) = conv2(extend_image,double(oa),'same');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
measure(1:rows,1:cols) = 0;

SSS = [];
temp = [];
for i = 1:rows
    for j = 1:cols
%         ttt = [];
%         for direction=1:p
%                     ttt = template(i+wid,j+wid,direction);
%         end
%         if (sum(abs(ttt))>0)
        
        Temp = [];
        Tem = [];
        for direction=1:p
            %%%%%%  USING 7*7 template
            a1 = template(i+wid-1,j+wid+3,direction);
            a2 = template(i+wid,j+wid+3,direction);
            a3 = template(i+wid+1,j+wid+3,direction);
                    
            a4 = template(i+wid-2,j+wid+2,direction);
            a5 = template(i+wid-1,j+wid+2,direction);
            a6 = template(i+wid,j+wid+2,direction);
            a7 = template(i+wid+1,j+wid+2,direction);
            a8 = template(i+wid+2,j+wid+2,direction);
                    
            a9 = template(i+wid-3,j+wid+1,direction);
            a10 = template(i+wid-2,j+wid+1,direction);
            a11 = template(i+wid-1,j+wid+1,direction);
            a12 = template(i+wid,j+wid+1,direction);
            a13 = template(i+wid+1,j+wid+1,direction);
            a14 = template(i+wid+2,j+wid+1,direction);
            a15 = template(i+wid+3,j+wid+1,direction);
                    
            a16 = template(i+wid-3,j+wid,direction);
            a17 = template(i+wid-2,j+wid,direction);
            a18 = template(i+wid-1,j+wid,direction);
            a19 = template(i+wid,j+wid,direction);
            a20 = template(i+wid+1,j+wid,direction);
            a21 = template(i+wid+2,j+wid,direction);
            a22 = template(i+wid+3,j+wid,direction);
                    
            a23 = template(i+wid-3,j+wid-1,direction);
            a24 = template(i+wid-2,j+wid-1,direction);
            a25 = template(i+wid-1,j+wid-1,direction);
            a26 = template(i+wid,j+wid-1,direction);
            a27 = template(i+wid+1,j+wid-1,direction);
            a28 = template(i+wid+2,j+wid-1,direction);
            a29 = template(i+wid+3,j+wid-1,direction);
                    
            a30 = template(i+wid-2,j+wid-2,direction);
            a31 = template(i+wid-1,j+wid-2,direction);
            a32 = template(i+wid,j+wid-2,direction);
            a33 = template(i+wid+1,j+wid-2,direction);
            a34 = template(i+wid+2,j+wid-2,direction);
                    
            a35 = template(i+wid-1,j+wid-3,direction);
            a36 = template(i+wid,j+wid-3,direction);
            a37 = template(i+wid+1,j+wid-3,direction);
            Tem = [Tem [a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 a11 a12 a13 a14 a15 a16 a17 a18 a19 a20...
                a21 a22 a23 a24 a25 a26 a27 a28 a29 a30 a31 a32 a33 a34 a35 a36 a37]'];

        end
        Temp = [Temp; Tem];

        M = (abs(Temp))'*(abs(Temp));
        E=abs(eig(M));
        
%         SSS = [SSS; E'];
        
%         NE = E./max(E);
        TT = det(M)/(trace(M)+eps);

%         KK = sum(NE);
     
        measure(i,j) = TT;
%         normalized(i,j) = KK;    
%         end
     
    end
end

% bb = max(measure(:))
marked_img(1:rows,1:cols) = 0;
II = extend_image(wid+1:rows+wid,wid+1:cols+wid);
[rr,cc,max_local] = findLocalMaximum(measure,10);
corner_num = 0;
N=[];
for i=1:size(rr,1)
    if ((measure(rr(i),cc(i))>threshold))
        corner_num = corner_num + 1;
        marked_img(rr(i),cc(i)) = 1;
%         KKK=measure(rr(i),cc(i));
%         N1=[KKK,rr(i),cc(i)];
%         N=[N;N1];
    end
end
% if isempty(N)
%    N2=[];
% else
% N2=sortrows(N,-1);
% end
%  locs=N2;

  
%  figure;
%  imshow(II,[]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

function img1=mark(img,x,y,w)
x = round(x);
y = round(y);

[M,N,C]=size(img);
img1=img;

if isa(img,'logical')
    img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:) = ...
        (img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)<1);
    img1(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:)=...
        img(x-floor(w/2)+1:x+floor(w/2)-1,y-floor(w/2)+1:y+floor(w/2)-1,:);
else
    img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:) = ...
        (img1(max(1,x-floor(w/2)):min(M,x+floor(w/2)),max(1,y-floor(w/2)):min(N,y+floor(w/2)),:)<128)*255;
    img1(max(1,x+1-floor(w/2)):min(M,x-1+floor(w/2)),max(1,y+1-floor(w/2)):min(N,y-1+floor(w/2)),:) = ...
        img(max(1,x+1-floor(w/2)):min(M,x-1+floor(w/2)),max(1,y+1-floor(w/2)):min(N,y-1+floor(w/2)),:);
   
end
end

function [row,col,max_local] = findLocalMaximum(val,radius)
    
    mask  = fspecial('disk',radius)>0;
    nb    = sum(mask(:));
    highest          = ordfilt2(val, nb, mask);
    second_highest   = ordfilt2(val, nb-1, mask);
    index            = highest==val & highest~=second_highest;
    max_local        = zeros(size(val));
    max_local(index) = val(index);
    [row,col]        = find(index==1);


    % FIND UNIQUE LOCAL MAXIMA (FAST)
    % val_height  = size(val,1);
    % val_width   = size(val,2);
    % max_local   = zeros(val_height,val_width);
    % val_enlarge = zeros(val_height+2*radius,val_width+2*radius);
    % val_mask    = zeros(val_height+2*radius,val_width+2*radius);
    % val_enlarge( (1:val_height)+radius , (1:val_width)+radius ) = val;
    % val_mask(    (1:val_height)+radius , (1:val_width)+radius ) = 1;
    % mask  = fspecial('disk',radius)>0;
    % row = zeros(val_height*val_width,1);
    % col = zeros(val_height*val_width,1);
    % index = 0;
    % for l = 1:val_height
    %     for c = 1:val_width
    %         val_ref = val(l,c);
    %         neigh_val  = val_enlarge(l:l+2*radius,c:c+2*radius);
    %         neigh_mask = val_mask(   l:l+2*radius,c:c+2*radius).*mask;
    %         neigh_sort = sort(neigh_val(neigh_mask==1));
    %         if val_ref==neigh_sort(end) && val_ref>neigh_sort(end-1)
    %             index          = index+1;
    %             row(index,1)   = l;
    %             col(index,1)   = c;
    %             max_local(l,c) = val_ref;
    %         end
    %     end
    % end
    % row(index+1:end,:) = [];
    % col(index+1:end,:) = [];


end

function [measure,mark_map] = affine3(im)
im=double(im);
  g = rand(1,1)*10.0+5.0;
  affine_im=im+g*randn(size(im));
    
    [measure,marked_img]=improved3(affine_im);
 
   [~,marked_img1]=improved3(im);
   
     [num,mark_map] = match_corner2(marked_img,marked_img1);
     
     
%     theta1 = rotation(2,1);
%   rotation_matrix1 = [cos(theta1),-1*sin(theta1),0; sin(theta1), cos(theta1),0;0,0,1];
%     tmp_scale1 = scale(2,:);
%     scale_matrix1 = [tmp_scale1(1),0,0;0,tmp_scale1(2),0;0,0,1];
%     tmp_shear1 = shear(2);
%     shear_matrix1 = [1,0,0;tmp_shear1(1),1,0;0,0,1];
%     
%     affine_matrix1 = shear_matrix1*scale_matrix1*rotation_matrix1;
%     tform3 = affine2d(affine_matrix1');
%     tform4 = affine2d((inv(affine_matrix1))');
%     affine_im1 = imwarp(im,tform3,'linear','FillValues',0);
%     
%      [~,marked_img]=improved3(affine_im1);
%     marked_img1 = imwarp(marked_img,tform4,'linear','FillValues',0)>0.1;
%    
%      I1=resize1(marked_img1,xc);
%      [num,mark_map1] = match_corner2(I1,mark_map);
     
end

function [I2]=resize1(imge_map2,n)
[r,c]=size(n);
 [r2,c2]=size(imge_map2);
 if mod(c,2)==0 && mod(r,2)==0
 I2=imge_map2(round((r2)/2)-round((r)/2)+1:round((r2)/2)+round((r)/2),round((c2)/2)-round((c)/2)+1:round((c2)/2)+round((c)/2));
elseif mod(c,2)==0
   I2=imge_map2(round((r2)/2)-round((r-1)/2):round((r2)/2)+round((r-1)/2),round((c2)/2)-round((c)/2)+1:round((c2)/2)+round((c)/2));
elseif mod(r,2)==0
   I2=imge_map2(round((r2)/2)-round((r)/2)+1:round((r2)/2)+round((r)/2),round((c2)/2)-round((c-1)/2):round((c2)/2)+round((c-1)/2));
else
    I2=imge_map2(round((r2)/2)-round((r-1)/2):round((r2)/2)+round((r-1)/2),round((c2)/2)-round((c-1)/2):round((c2)/2)+round((c-1)/2));
 end
end
function [num,t] = match_corner2(I1, standard_corner_image)

[rows,cols] = size(standard_corner_image);
num=0;
width =4;
t=zeros(rows,cols);
tmp_detected_corner_image1 = zeros(rows+2*width,cols+2*width);
tmp_detected_corner_image1(width+1:end-width,width+1:end-width) = I1;
for i = 1:1:rows
    for j = 1:1:cols
         if 1 == standard_corner_image(i,j)
%              'is wrong ???'
            if sum(sum(tmp_detected_corner_image1(i:i+2*width,j:j+2*width))) > 0
               t(i,j)=1;
               num=num+1;   
            end
        end
    end
end
end