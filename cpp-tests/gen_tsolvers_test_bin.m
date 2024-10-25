function gen_tsolvers_test_bin(m, dim, nvec, n_iter, l, f, s, fname)

if (nargin < 8)
    fname = sprintf('test_tsolvers_%d_%d_%d_%d.bin', m, dim, nvec, n_iter);
end

coord  = rand(m, dim);
dnoise = rand(m, 1) * 0.01;

K = gaussian_kernel(coord, coord, l, f, s, dnoise);
Afun = @(x) (K * x);
Pfun = @(x) (x);       % No precond

B = randn(m, nvec);

mpcg_t = tic;
[mpcg_X, mpcg_T] = mpcg(Afun, Pfun, B, n_iter, zeros(m, nvec));
mpcg_t = toc(mpcg_t);
fprintf('mpcg() time  = %.3f s\n', mpcg_t);

mfom_t = tic;
[mfom_X, mfom_T] = mfom(Afun,       B, n_iter, zeros(m, nvec));
mfom_t = toc(mfom_t);
fprintf('mfom() time  = %.3f s\n', mfom_t);

bd_data = [l; f; s];
bd_data = [bd_data; reshape(coord, [m * dim, 1])];
bd_data = [bd_data; dnoise];
bd_data = [bd_data; reshape(B, [m * nvec, 1])];
bd_data = [bd_data; reshape(mpcg_X, [m * nvec, 1])];
for i = 1 : nvec
    bd_data = [bd_data; reshape(mpcg_T{i}, [n_iter * n_iter, 1])];
end
bd_data = [bd_data; reshape(mfom_X, [m * nvec, 1])];
for i = 1 : nvec
    bd_data = [bd_data; reshape(mfom_T{i}, [n_iter * n_iter, 1])];
end

write_binary(fname, bd_data, 'double');
fprintf('Binary data write to %s\n', fname);

end