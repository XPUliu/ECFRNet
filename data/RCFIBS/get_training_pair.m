clc;close all;

working_dir = './RCFIBS_datasets/';
save_dir= '../';
dataset_name = 'ECFRNet';
load([working_dir, 'RCFIBS.mat']);

training_number = 4000;
test_number = 128;
training_offset = 0;
total_number = training_number+test_number;


feature_dim = 2;

%generate random rotation
translation = (2*rand(total_number,2)-1)*4;

block_size=33;
block_center=16;


im = uint8(zeros(total_number,3,block_size,block_size));
warped_im = uint8(zeros(total_number,3,block_size,block_size));
rotated_im=uint8(zeros(total_number,3,block_size,block_size));
transform_matrix = zeros(total_number,feature_dim);
 transform_matrix1=zeros(total_number,feature_dim);
r=randperm(total_number);
patches_total=N1(r,:,:,:);
disp(size(patches_total))
parpool;
parfor i = 1:total_number
    I = patches_total(i,:,:,:);
    disp(size(I));
    I = squeeze(I);
    disp(size(I));
 
    tmp_translation = round(translation(i,:));


 %translated RCFIB
   
    I_center_x = round(size(I,2)/2);
    I_center_y = round(size(I,1)/2);

    J_center_x = round(size(I,2)/2)+tmp_translation(1);
    J_center_y = round(size(I,1)/2)+tmp_translation(2);

    %check boundary
    if J_center_x < block_center+1
        J_center_x = block_center+1;
        tmp_translation(1) = block_center + 1 - round(size(I,2)/2);
    end
    if J_center_x + block_center > size(I,2)
        J_center_x = size(I,2) - block_center;
        tmp_translation(1) = size(I,2) - block_center - round(size(I,2)/2);
    end

    if J_center_y < block_center+1
        J_center_y = block_center+1;
        tmp_translation(2) = block_center + 1 - round(size(I,1)/2);
    end
    if J_center_y + block_center > size(I,1)
        J_center_y = size(I,1) - block_center;
        tmp_translation(2) = size(I,1) - block_center - round(size(I,1)/2);
    end
 %RCFIBS 
    crop_I = I(I_center_y-16:I_center_y+16,...
        I_center_x-16:I_center_x+16,:);
   %translated RCFIBS
    crop_J = I(J_center_y-16:J_center_y+16,...
        J_center_x-16:J_center_x+16,:);
    
    %gt transform
    transform_matrix(i,:) = [tmp_translation(1)./(16*2/3.0),tmp_translation(2)./(16*2/3.0)];

    crop_I = permute(crop_I,[3,1,2]);
    im(i,:,:,:) = crop_I;
     disp(size(crop_I));
 
    crop_J = permute(crop_J,[3,1,2]);
    warped_im(i,:,:,:) = crop_J;
end
delete(gcp);

total_im = im;
disp(size(total_im ));
total_warped_im = warped_im;
disp(size(total_warped_im));
total_transform_matrix = transform_matrix;

im = total_im(1:training_number,:,:,:);
warped_im = total_warped_im(1:training_number,:,:,:);
transform_matrix = total_transform_matrix(1:training_number,:);
save([save_dir, 'train_pair/' dataset_name '_train_block.mat'],'im','warped_im','transform_matrix');

index = training_number+1;
im = total_im(index:(index+test_number-1),:,:,:);
warped_im = total_warped_im(index:(index+test_number-1),:,:,:);
transform_matrix = total_transform_matrix(index:(index+test_number-1),:);

save([save_dir, 'train_pair/', dataset_name '_test_block.mat'],'im','warped_im','transform_matrix');
%fclose(fout);
