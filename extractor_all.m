clear;clc;
datasets_name='test';
load_feature_name='Candidate_corner_prediction';
save_feature_name='corner_position';
point_number=1000;

addpath(genpath('./external/vlfeat-0.9.18/toolbox/mex/'));
addpath(genpath('./HardNet/'));
dir_name = './data/';

if strcmp(datasets_name, 'test')
    subsets = {'lab'};
elseif strcmp(datasets_name, 'WebcamDataset')
    subsets = {'Chamonix','Courbevoie','Frankfurt','Mexico','Panorama','StLouis'};
end

image_list = load_image_list([dir_name datasets_name '/'], datasets_name);
[s, mess, messid] = mkdir([dir_name save_feature_name '/' datasets_name '/']);

for set_index = 1:numel(subsets)
    subset = subsets{set_index};
    image_list = load_image_list([dir_name 'datasets/' datasets_name '/'], [subset '/']);
    [s, mess, messid] = mkdir([dir_name save_feature_name '/' datasets_name '/' subset '/']);

    for i = 1:numel(image_list)
        feature = [];
        score = [];
        descriptor =[];
        imagepath=[dir_name 'datasets/' datasets_name '/' subset '/' image_list(i).name];
        try
            disp(imagepath);
            x = load([dir_name '/' load_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat']);
        catch
            disp(image_list(i).name);
            disp('bad load');
            save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score','descriptor');
            continue;
        end
    
        if numel(x.output_list) == 0
            disp(image_list(i).name);
            disp('no source')
            save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score','descriptor');
            continue;
        end
        output = x.output_list;
        if numel(output)==0
            disp(image_list(i).name);
            save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score','descriptor');
            continue;
        end
    

            output = permute(output,[3,1,2]);
        
            if size(output,1)==0 
                break
            end
            output_t = output;
            output = zeros(2,size(output,2),size(output,3));

            output(1,:,:) = output_t(1,:,:);
            output(2,:,:) = output_t(2,:,:);

            
            radius_factor = 1;
          
            outputs = permute(output,[2,3,1]);
            outputs(:,:,1) = outputs(:,:,1)*16*2/3;
            outputs(:,:,2) = outputs(:,:,2)*16*2/3;
    
            output_width = size(outputs,2);
            output_height = size(outputs,1);
            
            grid_x = (1:output_width)';
            grid_x = repmat(grid_x,1,output_height);
            grid_x = grid_x';
    
            grid_y = (1:output_height)';
            grid_y = repmat(grid_y,1,output_width);
            grid_x = grid_x - outputs(:,:,1);
            grid_y = grid_y - outputs(:,:,2);
            
            vote = zeros(size(grid_x));
            for j = 1:size(grid_x,1)
                for k = 1:size(grid_x,2)
                    index_x = k-ceil(outputs(j,k,1)/2);
                    index_y = j-ceil(outputs(j,k,2)/2);
                    frac_x = ceil(outputs(j,k,1)/2) - outputs(j,k,1)/2;
                    frac_y = ceil(outputs(j,k,2)/2) - outputs(j,k,2)/2;
                    if (index_x)>=1&&(index_x+1)<=output_width&&(index_y)>=1&&(index_y+1)<=output_height
                        vote(index_y+1,index_x+1) = vote(index_y+1,index_x+1)+frac_x*frac_y;
                        vote(index_y+1,index_x) = vote(index_y+1,index_x)+frac_y*(1-frac_x);
                        vote(index_y,index_x+1) = vote(index_y,index_x+1)+(1-frac_y)*frac_x;
                        vote(index_y,index_x) = vote(index_y,index_x)+(1-frac_x)*(1-frac_y);
                    end
                end
            end
            [vote, binary_img] = ApplyNonMax2Score(vote);
            binary_img = binary_img.*(vote>=2.40);
            vote2=vote;
            vote = reshape(vote,1,output_width*output_height);
  
            grid_x = reshape(grid_x,1,output_width*output_height);
            grid_y = reshape(grid_y,1,output_width*output_height);
            real_output = reshape(outputs,output_width*output_height,2);
            binary_img = reshape(binary_img,1,output_width*output_height);
         
            vote(~binary_img) = [];
            grid_x(~binary_img) = [];
            grid_y(~binary_img) = [];
            real_output(~binary_img,:) = [];
            
            [~,idx] = sort(vote,'descend');
            
            grid_x = grid_x(idx(1:min(size(idx,2),round(point_number/radius_factor^2))));
            grid_y = grid_y(idx(1:min(size(idx,2),round(point_number/radius_factor^2))));
            real_output = real_output(idx(1:min(size(idx,2),round(point_number/radius_factor^2))),:);
            score = vote(idx(1:min(size(idx,2),round(point_number/radius_factor^2))))';
            if(isempty(grid_x))
                continue;
            end          
            real_output(:,1) = grid_x;
            real_output(:,2) = grid_y;
            
            desc=hard(imagepath,real_output);
            descriptor=desc;
            feature = real_output;
        disp([image_list(i).name(1:end-4) '.mat'])
      save([dir_name save_feature_name '/' datasets_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature','score','descriptor');
    end
  end 

