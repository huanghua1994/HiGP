function gen_kernels_test_bin(n0, n1, dim, l, fname)

if (nargin < 5)
    fname = sprintf('test_kernels_%d_%d_%d.bin', n0, n1, dim);
end

c0 = rand(n0, dim);
c1 = rand(n1, dim);

K_pdist2 = pdist2(c0, c1);
K_pdist2 = K_pdist2.^2;

K_gaussian = gaussian_kernel(c0, c1, l, 1.0, 0.0);
dKdl_gaussian = gaussian_dkernel(c0, c1, l, 1.0, 0.0, 'l');

K_matern32 = matern32_kernel(c0, c1, l, 1.0, 0.0);
dKdl_matern32 = matern32_dkernel(c0, c1, l, 1.0, 0.0, 'l');

K_matern52 = matern52_kernel(c0, c1, l, 1.0, 0.0);
dKdl_matern52 = matern52_dkernel(c0, c1, l, 1.0, 0.0, 'l');

bd_data = [l; reshape(c0, [n0 * dim, 1])];
bd_data = [bd_data; reshape(c1, [n1 * dim, 1])];
bd_data = [bd_data; reshape(K_pdist2, [n0 * n1, 1])];
bd_data = [bd_data; reshape(K_gaussian, [n0 * n1, 1])];
bd_data = [bd_data; reshape(dKdl_gaussian, [n0 * n1, 1])];
bd_data = [bd_data; reshape(K_matern32, [n0 * n1, 1])];
bd_data = [bd_data; reshape(dKdl_matern32, [n0 * n1, 1])];
bd_data = [bd_data; reshape(K_matern52, [n0 * n1, 1])];
bd_data = [bd_data; reshape(dKdl_matern52, [n0 * n1, 1])];

write_binary(fname, bd_data, 'double');
fprintf('Binary data write to %s\n', fname);

end