function check_test_precond_gpr(n_train, n_pred, l, f, s, tf_func, glr_k, fsai_k, n_iter, n_vec, max_iter, rel_tol)

X_fname  = sprintf('pgpr_X_%d-%d.bin', n_train, n_pred);
Y_fname  = sprintf('pgpr_Y_%d-%d.bin', n_train, n_pred);
Z_fname  = sprintf('lanquad_Z_%dx%d.bin', n_train, n_vec);
sd_fname = sprintf('pgpr_sd_%d.bin', n_pred);
X_all = read_bin_mat(X_fname, [n_train + n_pred, 1]);
Y_all = read_bin_mat(Y_fname, [n_train + n_pred, 1]);
Z     = read_bin_mat(Z_fname, [n_train, n_vec]);
C_sd  = read_bin_mat(sd_fname, [n_train, 1]);
C_L   = read_bin_mat('pgpr_L.bin', [4, 1]);

X_train  = X_all(1 : n_train, :);
Y_train  = Y_all(1 : n_train, :);
X_pred   = X_all(n_train + 1 : end, :);
C_Y_pred = Y_all(n_train + 1 : end, :);

npt_s = -glr_k-1;  % Prevent AFN from falling back to Nystrom
[L, L_grad] = precond_gpr_loss(X_train, Y_train, l, s, f, tf_func, npt_s, ...
                               glr_k, fsai_k, n_iter, n_vec, zeros(3, 1), Z);

print_scalar_error('L        ', L,         C_L(1));
print_scalar_error('L_grad(l)', L_grad(1), C_L(2));
print_scalar_error('L_grad(f)', L_grad(3), C_L(3));
print_scalar_error('L_grad(s)', L_grad(2), C_L(4));

[Y_pred, stddev] = precond_gpr_predict([l, s, f], tf_func, X_train, Y_train, X_pred, ...
                                       npt_s, glr_k, fsai_k, max_iter, rel_tol);
relerr_Y_pred = norm(Y_pred - C_Y_pred) / norm(Y_pred);
relerr_stddev = norm(stddev - C_sd)     / norm(stddev);
fprintf('Y_pred and stddev relerr = %.2e, %.2e\n\n', relerr_Y_pred, relerr_stddev);

end

function print_scalar_error(name, ref_val, val)
fprintf('%s = % .6e, relerr = %.6e\n', name, ref_val, abs((ref_val - val) / ref_val));
end