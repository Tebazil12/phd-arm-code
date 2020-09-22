figure(2)
clf
hold on
plot(ex.taps_disp_mu_preds(:,1), ex.taps_disp_mu_preds(:,2),'x')

x = model.x_gplvm_input_train(1+n_ref_taps:end,1);
y = model.x_gplvm_input_train(1+n_ref_taps:end,2);

plot(x,y,'-o')

grid on