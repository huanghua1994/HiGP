function gen_afn_test_bin(m, dim, nvec, glr_rank, fsai_npt, l, f, s, fname)

if (nargin < 9)
    fname = sprintf('test_afn_%d_%d_%d_%d_%d.bin', m, dim, nvec, glr_rank, fsai_npt);
end

kernel = @(X, Y, dnoise) gaussian_kernel(X, Y, l, f, s, dnoise);
n_grad = 3;
dkernels = cell(n_grad, 1);
dkernels{1} = @(X, Y) gaussian_dkernel(X, Y, l, f, s, 'l');
dkernels{2} = @(X, Y) gaussian_dkernel(X, Y, l, f, s, 's');
dkernels{3} = @(X, Y) gaussian_dkernel(X, Y, l, f, s, 'f');

param  = [l, s, f];
coord  = rand(m, dim);
dnoise = rand(m, 1) * 0.01;
npt_s  = -glr_rank-1;  % Mute rank estimation, always uses AFN instead of Nystrom
build_t = tic;
ap = afn_build(kernel, param, dnoise, coord, npt_s, glr_rank, fsai_npt, n_grad, dkernels);
build_t = toc(build_t);
fprintf('afn_build() time  = %.3f s\n', build_t);

gt = afn_grad_trace(ap);
B = randn(m, nvec);

apply_t = tic;
C = afn_apply(ap, B);
apply_t = toc(apply_t);
fprintf('afn_apply() time  = %.3f s\n', apply_t);

dapply_t = tic;
D = afn_dapply(ap, B);
dapply_t = toc(dapply_t);
fprintf('afn_dapply() time = %.3f s\n', dapply_t);

bd_data = [l; f; s];
bd_data = [bd_data; reshape(coord, [m * dim, 1])];
bd_data = [bd_data; dnoise];
bd_data = [bd_data; gt(1); gt(3); gt(2); ap.logdet];
bd_data = [bd_data; reshape(B, [m * nvec, 1])];
bd_data = [bd_data; reshape(C, [m * nvec, 1])];
bd_data = [bd_data; reshape(D{1}, [m * nvec, 1])];
bd_data = [bd_data; reshape(D{3}, [m * nvec, 1])];
bd_data = [bd_data; reshape(D{2}, [m * nvec, 1])];

write_binary(fname, bd_data, 'double');
fprintf('Binary data write to %s\n', fname);

end