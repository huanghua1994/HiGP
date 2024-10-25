function check_test_exact_gpr(n_train, n_pred, l, f, s, tf_func)

X_fname  = sprintf('egp_X_%d-%d.bin', n_train, n_pred);
Y_fname  = sprintf('egp_Y_%d-%d.bin', n_train, n_pred);
sd_fname = sprintf('egp_sd_%d.bin', n_pred);
X_all = read_bin_mat(X_fname, [n_train + n_pred, 1]);
Y_all = read_bin_mat(Y_fname, [n_train + n_pred, 1]);
C_sd  = read_bin_mat(sd_fname, [n_train, 1]);
C_L   = read_bin_mat('egpr_L.bin', [4, 1]);

X_train  = X_all(1 : n_train, :);
Y_train  = Y_all(1 : n_train, :);
X_pred   = X_all(n_train + 1 : end, :);
C_Y_pred = Y_all(n_train + 1 : end, :);

[L, L_grad] = exact_gpr_loss(X_train, Y_train, l, s, f, tf_func);

print_scalar_error('L        ', L,         C_L(1));
print_scalar_error('L_grad(l)', L_grad(1), C_L(2));
print_scalar_error('L_grad(f)', L_grad(3), C_L(3));
print_scalar_error('L_grad(s)', L_grad(2), C_L(4));

[Y_pred, stddev] = exact_gpr_predict([l, s, f], tf_func, X_train, Y_train, X_pred);
relerr_Y_pred = norm(Y_pred - C_Y_pred) / norm(Y_pred);
relerr_stddev = norm(stddev - C_sd)     / norm(stddev);
fprintf('Y_pred and stddev relerr = %.2e, %.2e\n\n', relerr_Y_pred, relerr_stddev);

end

function print_scalar_error(name, ref_val, val)
fprintf('%s = % .6e, relerr = %.6e\n', name, ref_val, abs((ref_val - val) / ref_val));
end