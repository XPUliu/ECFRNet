function d=hard(imagepath,a)
%% reload
% clear classes
% obj = py.importlib.import_module('myfun');
% py.importlib.reload(obj);
%%
im=imread(imagepath);

wid=17;
if size(im,3)==3
    im=rgb2gray(im);
end
[rows,cols]=size(im);
extend_image = zeros(rows+2*wid,cols+2*wid);
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
im1=uint8(extend_image);


if size(a,1)==0
    d=[];
else
a=floor(a);
for i=1:size(a,1)
    if a(i,1)==0
        a(i,1)=a(i,1)+1;
    end
    if a(i,2)==0
        a(i,2)=a(i,2)+1;
    end
            
    patch(i,1,:,:)=double(im1(a(i,2)+wid-16:a(i,2)+wid+15,a(i,1)+wid-16:a(i,1)+wid+15,:));
end

b=mat2nparray(patch);
obj = py.importlib.import_module('HardNet'); 
 py.importlib.reload(obj)
d=obj.main (b);
d=nparray2mat(d);
d=d';

end