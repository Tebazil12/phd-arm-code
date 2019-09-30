load('/home/lizzie/git/masters-tactile/data/robotcode_first_adapt2019-02-15_1547/all_data.mat') %banana

252
clf
dissims=[];
ref = [ex.ref_diffs_norm(ex.ref_diffs_norm_max_ind ,:  ,1) ex.ref_diffs_norm(ex.ref_diffs_norm_max_ind ,:  ,2)];
for i = 1:63
    
    differences = ref - model.y_gplvm_input_train(i,:); 


    diss = norm([differences(:,1:126)'; differences(:,127:end)']);
    dissims =[dissims; diss]; %#ok<AGROW>
end

hold on 
for i = 1:3
    1+(i-1)*21
    i*21
plot3( model.x_gplvm_input_train(1+(i-1)*21:i*21,1),...
       model.x_gplvm_input_train(1+(i-1)*21:i*21,2),...
       dissims(1+(i-1)*21:i*21))
end

xlabel("Estimated displacement / mm")
ylabel("Predicted \mu")
zlabel("Dissimilarity")
title("GPLVM Model")