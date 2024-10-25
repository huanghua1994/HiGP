function check_test_precond_gpc(n_train, n_pred, n_class, n_sample, tf_func, glr_k, fsai_k, n_iter, n_vec, max_iter, rel_tol)

X_fname  = sprintf('pgpc_X_%d-%d.bin', n_train, n_pred);
Y_fname  = sprintf('pgpc_Y_%d-%d.bin', n_train, n_pred);
Z_fname  = sprintf('lanquad_Z_%dx%d.bin', n_train, n_vec);
p_fname  = sprintf('pgpc_params_%d.bin', n_class);
L_fname  = sprintf('pgpc_L_%d.bin', n_class);
Yc_fname = sprintf('pgpc_Yc_%d-%d.bin', n_pred, n_class);
pb_fname = sprintf('pgpc_pb_%d-%d.bin', n_pred, n_class);
rv_fname = sprintf('pgpc_rndvec_%dx%dx%d.bin', n_class, n_pred, n_sample);

X_all  = read_bin_mat(X_fname,  [n_train + n_pred, 1]);
Y_all  = read_bin_mat(Y_fname,  [n_train + n_pred, 1], 'int') + 1;  % Convert to 1-based index
Z      = read_bin_mat(Z_fname,  [n_train, n_vec]);
params = read_bin_mat(p_fname,  [n_class, 3]);
Lgrads = read_bin_mat(L_fname,  [3 * n_class + 1, 1]);
C_Yc   = read_bin_mat(Yc_fname, [n_pred, n_class]);
C_pb   = read_bin_mat(pb_fname, [n_pred, n_class]);
rndvec = read_bin_mat(rv_fname, [n_class * n_pred * n_sample, 1]);

X_train  = X_all(1 : n_train, :);
Y_train  = Y_all(1 : n_train, :);
X_pred   = X_all(n_train + 1 : end, :);
C_Y_pred = Y_all(n_train + 1 : end, :);
C_L       = Lgrads(1);
C_L_grads = reshape(Lgrads(2 : end), [n_class, 3]);
ls = params(:, 1);
fs = params(:, 2);
ss = params(:, 3);

[dnoises, Ys] = gpc_process_label(Y_train);

npt_s = -glr_k-1;  % Prevent AFN from falling back to Nystrom
[L, L_grads] = precond_gpc_loss(X_train, Ys, dnoises, ls, ss, fs, tf_func, npt_s, ...
                                glr_k, fsai_k, n_iter, n_vec, zeros(3, 1), Z);
L_grads = reshape(L_grads, [n_class, 3]);
fprintf('Loss   relerr = %e\n', abs(L - C_L) / abs(L));
fprintf('l grad relerr = %e\n', norm(L_grads(:, 1) - C_L_grads(:, 1)) / norm(L_grads(:, 1)));
fprintf('f grad relerr = %e\n', norm(L_grads(:, 3) - C_L_grads(:, 2)) / norm(L_grads(:, 3)));
fprintf('s grad relerr = %e\n', norm(L_grads(:, 2) - C_L_grads(:, 3)) / norm(L_grads(:, 2)));
fprintf('\n');

hp = [ls; ss; fs];
[Y_pred, Y_pred_c, probab] = precond_gpc_predict(...
    hp, tf_func, X_train, Y_train, X_pred, ...
    n_sample, npt_s, glr_k, fsai_k, max_iter, rel_tol, rndvec ...
);
fprintf('Y_pred   relerr = %e\n', norm(Y_pred   - C_Y_pred) / norm(Y_pred));
fprintf('Y_pred_c relerr = %e\n', norm(Y_pred_c - C_Yc)     / norm(Y_pred_c));
fprintf('probab   relerr = %e\n', norm(probab   - C_pb)     / norm(probab));
fprintf('\n');

end
    