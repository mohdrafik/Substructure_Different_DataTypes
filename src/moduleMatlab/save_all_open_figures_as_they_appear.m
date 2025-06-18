
save_path = input("enter the save path : ");
save_folder = save_path;
save_all_open_figures_screenshot_style(save_folder);



function save_all_open_figures_screenshot_style(save_folder)
    if ~exist(save_folder, 'dir')
        mkdir(save_folder);
    end

    figHandles = findall(0, 'Type', 'figure');

    for i = 1:length(figHandles)
        fig = figHandles(i);
        figure(fig);  % Bring to front

        % Capture frame (visible appearance only)
        frame = getframe(fig);

        % Save image
        img = frame.cdata;
        filename = fullfile(save_folder, sprintf('Figure_%d_screenshot22.png', fig.Number));
        imwrite(img, filename);

        fprintf("Saved screenshot-like figure: %s\n", filename);
    end
end


