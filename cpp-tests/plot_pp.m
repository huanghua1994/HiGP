function plot_pp(bin_fname, root_enbox_size, dim, pp_sizes)

fid = fopen(bin_fname, 'rb');
pp_data = fread(fid, [dim * sum(pp_sizes), 1], 'double');
fclose(fid);

cnt = 1;
curr_enbox_size = root_enbox_size;
for i = 1 : length(pp_sizes)
    pp_size_i = pp_sizes(i);
    fprintf('Level %d has %d proxy points\n', i, pp_size_i);
    if (pp_size_i == 0)
        curr_enbox_size = curr_enbox_size * 0.5;
        continue;
    end
    idx = cnt : (cnt + pp_size_i * dim - 1);
    pp_i = pp_data(idx);
    pp_i = reshape(pp_i, [pp_size_i, dim]);

    semi_enbox_size = 0.5 * curr_enbox_size;

    figure;
    if (dim == 2)
        Xp = rand(400, dim) .* curr_enbox_size - semi_enbox_size;
        scatter(Xp(:, 1), Xp(:, 2), 'r.'), hold on
        scatter(pp_i(:, 1), pp_i(:, 2), 'b*');
        xlabel('x'); ylabel('y');
        legend({'Xp points', 'Proxy Points'}, 'Location', 'eastoutside');
    end
    if (dim == 3)
        Xp = rand(1000, dim) .* curr_enbox_size - semi_enbox_size;
        scatter3(Xp(:, 1), Xp(:, 2), Xp(:, 3), 'r.'), hold on
        scatter3(pp_i(:, 1), pp_i(:, 2), pp_i(:, 3), 'b*');
        xlabel('x'); ylabel('y'); zlabel('z');
        legend({'Xp points', 'Proxy Points'}, 'Location', 'eastoutside');
    end
    grid on;
    title(sprintf('Level %d Proxy Points', i));
    cnt = cnt + pp_size_i * dim;

    curr_enbox_size = curr_enbox_size * 0.5;
end

end