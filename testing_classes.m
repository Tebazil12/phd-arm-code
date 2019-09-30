model = GPLVM
model.set_new_hyper_params(y_gplvm_input_train,x_gplvm_input_train)
model
model.sigma_f

current_step = 3;

for current_step = current_step+1:5
    current_step
   disp(".") 
end

save('') %this saves everything in workspace to /home/lizzie/git/masters-tactile/matlab.mat

delete(model.e)

model.e