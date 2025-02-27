clc; clear; close all;

% **Define Paths**
input_dir = "C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\data\intermdata1";
output_dir = "C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\plots";

% **Ensure Output Directory Exists**
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% **Get List of .MAT Files**
mat_files = dir(fullfile(input_dir, "*.mat"));

% **Loop Through Each .MAT File**
for i = 1:length(mat_files)
    % Load Data
    file_path = fullfile(input_dir, mat_files(i).name);
    fprintf("Processing: %s\n", mat_files(i).name);
    
    data_struct = load(file_path); % Load .MAT file
    field_names = fieldnames(data_struct); % Get variable names
    
    if isempty(field_names)
        fprintf("⚠ Warning: No valid data in %s. Skipping...\n", mat_files(i).name);
        continue;
    end

    % Assume the first variable is the volume data
    vol_data = data_struct.(field_names{1});
    
    % **Normalize Data (0 to 1)**
    vol_data = (vol_data - min(vol_data(:))) / (max(vol_data(:)) - min(vol_data(:)));

    % **Create UI Figure & Use uipanel for volshow**
    fig = uifigure('Name', mat_files(i).name, 'Color', 'black');
    panel = uipanel(fig, 'Position', [20 20 600 600]); % Panel for volshow

    % **Apply Color Grading (Brighter Inner, Lighter Outer)**
    cmap = hot(256); % Red-Yellow-White color gradient
    vol = volshow(vol_data, 'Parent', panel, 'Colormap', cmap, 'Renderer', 'VolumeRendering', ...
        'BackgroundColor', 'black', 'Alphamap', linspace(0.05, 1, 256));

    % **Save as GIF**
    gif_filename = fullfile(output_dir, replace(mat_files(i).name, '.mat', '.gif'));
    
    % Capture Frame
    frame = getframe(panel);
    imwrite(frame2im(frame), gif_filename, 'gif', 'LoopCount', inf, 'DelayTime', 0.1);
    
    fprintf("✅ Saved GIF: %s\n", gif_filename);
    
    % **Close UI Figure to Free Memory**
    close(fig);
end

fprintf("\n🎉✅ All .MAT files processed and saved as GIFs!\n");
