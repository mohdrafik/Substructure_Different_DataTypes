
clc; clear; close all;

% Root directory containing all the cluster folders
root_dir = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\hybrid_Kdbcluster';
set_threshold = 1.34;  % Eliminate background/noise

clusters = dir(fullfile(root_dir, 'cluster_*'));

for i = 1:length(clusters)
    cluster_path = fullfile(root_dir, clusters(i).name);
    mat_files = dir(fullfile(cluster_path, '*.mat'));

    for j = 1:length(mat_files)
        mat_file = fullfile(cluster_path, mat_files(j).name);
        fprintf('📂 Processing: %s\n', mat_file);

        % Load .mat file
        data = load(mat_file);
        field = fieldnames(data);
        coords = data.(field{1});  % Expecting [x y z intensity]

        % Thresholding based on intensity
        mask = coords(:, 4) >= set_threshold;
        x = coords(mask, 1);
        y = coords(mask, 2);
        z = coords(mask, 3);
        intensity = coords(mask, 4);

        fprintf("Filtered points: %d → %d (after thresholding)\n", size(coords, 1), sum(mask));

        % ------------------ Mesh/Point Plot with Alpha Shape ------------------
        fig = figure('Visible', 'on');
        hold on;
        alpha_value = 10;

        try
            shp = alphaShape(x, y, z, alpha_value);
            [tri, pts] = boundaryFacets(shp);

            if isempty(tri)
                fprintf("⚠ Cluster mesh empty for %s\n", mat_files(j).name);
                continue;
            end

            % Draw smooth mesh
            p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
            p.FaceAlpha = 0.6;
            p.EdgeColor = 'none';
            p.FaceColor = [0.2 + rand * 0.7, 0.5 + rand * 0.4, 1.0];

            % Overlay point cloud with transparency
            scatter3(x, y, z, 5, intensity, 'filled', 'MarkerFaceAlpha', 0.06);

        catch ME
            fprintf("⚠ Mesh creation failed: %s\n", ME.message);
            continue;
        end

        axis equal; grid on; view(3);
        colormap(parula); colorbar;
        title(mat_files(j).name, 'Interpreter', 'none');
        xlabel('X'); ylabel('Y'); zlabel('Z');

        % Save mesh plot as PNG
        [~, name, ~] = fileparts(mat_files(j).name);
        saveas(fig, fullfile(cluster_path, [name '_meshplot.png']));

        % ------------------ Create Rotating GIF ------------------
        gif_name = fullfile(cluster_path, [name '_rotating.gif']);
        nFrames = 36;
        for angle = linspace(0, 360, nFrames)
            view(angle, 30); drawnow;
            frame = getframe(fig);
            [imind, cm] = rgb2ind(frame2im(frame), 256);
            if angle == 0
                imwrite(imind, cm, gif_name, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
            else
                imwrite(imind, cm, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
            end
        end

        close(fig);

        % ------------------ Volumetric Rendering ------------------
        try
            % Construct volumetric grid
            grid_size = ceil(max(coords(:,1:3))) + 1;
            vol = accumarray(round(coords(:,1:3)), coords(:,4), grid_size, @mean, 0);

            % Normalize for visibility
            vol_norm = (vol - min(vol(:))) / (max(vol(:)) - min(vol(:)) + eps);
            vol_norm = permute(vol_norm, [2 1 3]);  % Adjust axes for volshow

            figVol = uifigure('Visible', 'on');
            volshow(vol_norm, ...
                'Colormap', parula, ...
                'Alphamap', linspace(0, 1, 256).^0.8, ...
                'BackgroundColor', 'black', ...
                'Renderer', 'VolumeRendering');

            % Export volume view
            exportgraphics(figVol, fullfile(cluster_path, [name '_volumeview.png']));
            close(figVol);

        catch ME
            fprintf("⚠ Volume rendering skipped for %s: %s\n", name, ME.message);
        end
    end
end

fprintf("\n✅ All mesh plots, rotating GIFs, and volumetric renderings generated!\n");



%%  <---------------- this below code is working --------------------------------------- >

% clc; clear; close all;
% 
% % Root directory containing all the cluster folders
% root_dir = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\hybrid_Kdbcluster';  % CHANGE this path to your actual one
% set_threshold = 1.34; % to eliminate the background.
% 
% clusters = dir(fullfile(root_dir, 'cluster_*'));
% 
% for i = 1:length(clusters)
%     cluster_path = fullfile(root_dir, clusters(i).name);
%     mat_files = dir(fullfile(cluster_path, '*.mat'));
% 
%     for j = 1:length(mat_files)
%         mat_file = fullfile(cluster_path, mat_files(j).name);
%         fprintf('📂 Processing: %s\n', mat_file);
% 
%         % Load .mat file
%         data = load(mat_file);
%         field = fieldnames(data);
%         coords = data.(field{1});  % [x y z intensity]
% 
% %         x = coords(:,1); y = coords(:,2); z = coords(:,3);
% %         intensity = coords(:,4);
%         threshold = set_threshold;
%         mask = coords(:,4) >= threshold;
%         
%         % Filtered values
%         x = coords(mask,1); 
%         y = coords(mask,2); 
%         z = coords(mask,3);
%         intensity = coords(mask,4);
% 
% fprintf("Filtered points: %d → %d (after thresholding)\n", size(coords,1), sum(mask));
% 
%         % ------------------ Create Mesh or Scatter Plot ------------------
%         fig = figure('Visible',' on');
%         try
%             tri = delaunay(x, y, z);
%             trisurf(tri, x, y, z, intensity, 'EdgeColor', 'none');
%             shading interp;
%         catch
%             scatter3(x, y, z, 10, intensity, 'filled');
%         end
% 
%         axis equal; grid on; view(3);
%         colormap jet;
%         colorbar;
%         title(mat_files(j).name, 'Interpreter', 'none');
%         xlabel('X'); ylabel('Y'); zlabel('Z');
% 
%         % Save PNG snapshot
%         [~, name, ~] = fileparts(mat_files(j).name);
%         saveas(fig, fullfile(cluster_path, [name '_meshplot.png']));
% 
%         % ------------------ Create Rotating GIF ------------------
%         gif_name = fullfile(cluster_path, [name '_rotating.gif']);
%         nFrames = 36;
%         for angle = linspace(0, 360, nFrames)
%             view(angle, 30);
%             drawnow;
%             frame = getframe(fig);
%             [imind, cm] = rgb2ind(frame2im(frame), 256);
%             if angle == 0
%                 imwrite(imind, cm, gif_name, 'gif', 'Loopcount', inf, 'DelayTime', 0.1);
%             else
%                 imwrite(imind, cm, gif_name, 'gif', 'WriteMode', 'append', 'DelayTime', 0.1);
%             end
%         end
% 
%         close(fig);
% 
%         % ------------------ Optional Volumetric View ------------------
%         % Convert to 3D volume grid (use only if dense enough)
%         try
%             vol = accumarray(round(coords(:,1:3)), coords(:,4), [], @mean, 0);
%             vol = permute(vol, [2 1 3]);
%             figVol = uifigure('Visible', ' on');
%             volshow(vol, 'Renderer', 'VolumeRendering');
%             exportgraphics(figVol, fullfile(cluster_path, [name '_volumeview.png']));
%             close(figVol);
%         catch
%             fprintf("⚠️ Volume rendering skipped for %s (possibly sparse)\n", name);
%         end
%     end
% end
% 
% fprintf("\n✅ All plots and rotating GIFs generated!\n");
% 


% clc; clear; close all;
% 
% % Set your root directory path
% root_dir = 'E:\YourPath\hybrid_Kdbcluster'; % Change to your actual path

% Get all cluster folders
% clusters = dir(fullfile(root_dir, 'cluster_*'));
% 
% % Loop through each cluster folder
% for i = 1:length(clusters)
%     cluster_path = fullfile(root_dir, clusters(i).name);
%     mat_files = dir(fullfile(cluster_path, '*.mat'));
%     
%     for j = 1:length(mat_files)
%         mat_file = fullfile(cluster_path, mat_files(j).name);
%         fprintf('📂 Processing: %s\n', mat_file);
% 
%         % Load the matrix (assumes first variable is the one we need)
%         data = load(mat_file);
%         fieldname = fieldnames(data);
%         coords = data.(fieldname{1});  % Expected size: [N x 4]
% 
%         % Extract coordinates and intensity
%         x = coords(:,1);
%         y = coords(:,2);
%         z = coords(:,3);
%         intensity = coords(:,4);
% 
%         % Try to triangulate the data for mesh
%         try
%             tri = delaunay(x, y, z);
% %             figure('Visible','off'); % Set to 'on' if needed
%             figure('Visible','on'); % Set to 'on' if needed
%             trisurf(tri, x, y, z, intensity, 'EdgeColor', 'none');
%             shading interp;
%         catch
%             % Fallback to scatter if mesh can't be built
%             figure('Visible','on'); % Set to 'on' if needed
% %             figure('Visible','off');
%             scatter3(x, y, z, 10, intensity, 'filled');
%         end
% 
%         % Visualization settings
%         title(strrep(mat_files(j).name, '_', '\_'), 'Interpreter', 'none');
%         xlabel('X'); ylabel('Y'); zlabel('Z');
%         colormap jet;
%         colorbar;
%         axis equal; grid on; view(3);
% 
%         % Save plot
%         [~, name, ~] = fileparts(mat_files(j).name);
%         saveas(gcf, fullfile(cluster_path, [name '_meshplot.png']));
%         pause(5);
%         close(gcf);
%     end
% end
% 
% fprintf('\n✅ All mesh plots generated and saved!\n');
% 
% 
% 




% 
% % List all folders inside root_dir starting with 'cluster_'
% clusters = dir(fullfile(root_dir, 'cluster_*'));
% 
% % Loop through each cluster folder
% for i = 1:length(clusters)
%     cluster_folder = fullfile(root_dir, clusters(i).name);
%     
%     % Get all .mat files in the cluster folder
%     mat_files = dir(fullfile(cluster_folder, '*.mat'));
%     
%     for j = 1:length(mat_files)
%         mat_path = fullfile(cluster_folder, mat_files(j).name);
%         fprintf('📂 Processing: %s\n', mat_path);
% 
%         % Load .mat file
%         data = load(mat_path);
%         fieldname = fieldnames(data);
%         coords = data.(fieldname{1});  % assuming the 1000x4 matrix is the first field
% 
%         % Extract x, y, z, intensity
%         x = coords(:,1);
%         y = coords(:,2);
%         z = coords(:,3);
%         intensity = coords(:,4);
% 
%         % Plot
%         figure('Visible','off');  % Set to 'on' if you want to see it
%         scatter3(x, y, z, 15, intensity, 'filled');
%         colormap jet;
%         colorbar;
%         title(sprintf('Mesh Plot: %s', mat_files(j).name), 'Interpreter', 'none');
%         xlabel('X'); ylabel('Y'); zlabel('Z');
%         axis equal; view(3); grid on;
% 
%         % Save figure as PNG in same cluster folder
%         [~, name, ~] = fileparts(mat_files(j).name);
%      
%         saveas(gcf, fullfile(cluster_folder, [name '_meshplot.png']));
% %         close(gcf);
%         
%     end
% end
% 
% fprintf('\n All cluster .mat files processed and plots saved.\n');
