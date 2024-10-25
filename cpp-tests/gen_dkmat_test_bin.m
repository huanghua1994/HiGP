function gen_dkmat_test_bin(n0, n1, dim, nvec, l, f, s, fname)

if (nargin < 8)
    fname = sprintf('test_dkmat_%d_%d_%d_%d.bin', n0, n1, dim, nvec);
end

c0 = rand(n0, dim);
c1 = rand(n1, dim);
dnoise = rand(n0, 1) * 0.01;

K    = gaussian_kernel(c0, c1, l, f, s, dnoise);
dKdl = gaussian_dkernel(c0, c1, l, f, s, 'l');
dKdf = gaussian_dkernel(c0, c1, l, f, s, 'f');
dKds = gaussian_dkernel(c0, c1, l, f, s, 's');

B = randn(n1, nvec);
K_B    = K    * B;
dKdl_B = dKdl * B;
dKdf_B = dKdf * B;
dKds_B = dKds * B;

bd_data = [l; f; s];
bd_data = [bd_data; reshape(c0, [n0 * dim, 1])];
bd_data = [bd_data; reshape(c1, [n1 * dim, 1])];
bd_data = [bd_data; dnoise];
bd_data = [bd_data; reshape(B, [n1 * nvec, 1])];
bd_data = [bd_data; reshape(K_B, [n0 * nvec, 1])];
bd_data = [bd_data; reshape(dKdl_B, [n0 * nvec, 1])];
bd_data = [bd_data; reshape(dKdf_B, [n0 * nvec, 1])];
bd_data = [bd_data; reshape(dKds_B, [n0 * nvec, 1])];

write_binary(fname, bd_data, 'double');
fprintf('Binary data write to %s\n', fname);

end