function gen_nys_test_bin(m, dim, nvec, nys_k, l, f, s, fname)

if (nargin < 8)
    fname = sprintf('test_nys_%d_%d_%d_%d.bin', m, dim, nvec, nys_k);
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
perm   = 1 : m;  % For convenience, we use the first nys_k points as the anchor points --> no permutation
build_t = tic;
np = nys_build(kernel, param, dnoise, coord, nys_k, perm, n_grad, dkernels);
build_t = toc(build_t);
fprintf('nys_build() time  = %.3f s\n', build_t);

gt = nys_grad_trace(np);
B = randn(m, nvec);

apply_t = tic;
C = nys_apply(np, B);
apply_t = toc(apply_t);
fprintf('nys_apply() time  = %.3f s\n', apply_t);

dapply_t = tic;
D = nys_dapply(np, B);
dapply_t = toc(dapply_t);
fprintf('nys_dapply() time = %.3f s\n', dapply_t);

bd_data = [l; f; s];
bd_data = [bd_data; reshape(coord, [m * dim, 1])];
bd_data = [bd_data; dnoise];
bd_data = [bd_data; gt(1); gt(3); gt(2); np.logdet];
bd_data = [bd_data; reshape(B, [m * nvec, 1])];
bd_data = [bd_data; reshape(C, [m * nvec, 1])];
bd_data = [bd_data; reshape(D{1}, [m * nvec, 1])];
bd_data = [bd_data; reshape(D{3}, [m * nvec, 1])];
bd_data = [bd_data; reshape(D{2}, [m * nvec, 1])];

write_binary(fname, bd_data, 'double');
fprintf('Binary data write to %s\n', fname);

end