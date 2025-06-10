%  because for cluster zero which are the masked values with zeros, and in
%  my data as well as measuring instruments the sample is totally
%  surrounded by the water (1.334 masked with zero here in my data), so
%  it's shows cube as well as the surface surrounding the sample also.
%  which seems to be as main data plot, but it's surface of water
%  refractive index surrounding the sample (main data).,


%% 
% % usage: % All clusters from each file in subplots (one figure per file)
% clusterlabelCoordPlot_v2('C:\path\to\your\matfiles', 'mesh', 5, 10, 0.6, true)
% 
% % One figure per cluster instead
% clusterlabelCoordPlot_v2('C:\path\to\your\matfiles', 'mesh+point', 4, 8, 0.5, false)


function clusterlabelCoordPlot_v2(data_dir, mode, alpha_value, min_cluster_size, face_transparency, use_subplots)
% =========================================================================
% 📊 clusterlabelCoordPlot_v2 - Visualize Labeled 3D Cluster Meshes from Files
% =========================================================================
% This function visualizes and saves 3D cluster meshes (via alphaShape)
% from coordinate-label .mat files (with 4 columns: x, y, z, label).
%
% 🔧 PARAMETERS:
% -------------------------------------------------------------------------
% data_dir            : Directory containing .mat files [x y z label]
% mode                : 'mesh', 'point', or 'mesh+point'
% alpha_value         : Alpha shape parameter (higher → smoother)
% min_cluster_size    : Minimum number of points to consider a cluster 10
% or less like 3,4,..

% face_transparency   : Surface transparency (0=transparent, 1=solid)
% use_subplots        : true = all clusters in one figure | false = separate figs
% =========================================================================

if nargin < 6, use_subplots = false; end
if nargin < 5, face_transparency = 0.6; end
if nargin < 4, min_cluster_size = 10; end
if nargin < 3, alpha_value = 5; end
if nargin < 2, mode = 'mesh'; end
if nargin < 1 || isempty(data_dir)
    error('Please provide a directory containing .mat files.');
end

custom_output_dir = fullfile(data_dir, 'visualization_results_mesh');
if ~exist(custom_output_dir, 'dir')
    mkdir(custom_output_dir);
end

mat_files = dir(fullfile(data_dir, '*.mat'));

for i = 1:length(mat_files)
    mat_path = fullfile(data_dir, mat_files(i).name);
    fprintf("\nProcessing: %s\n", mat_path);

    try
        alldata = load(mat_path);
        if isstruct(alldata)
            fieldNames = fieldnames(alldata);
            datavalues = alldata.(fieldNames{1});
        else
            datavalues = alldata;
        end

        coords = datavalues(:, 1:3);
        labels = datavalues(:, 4);
        unique_labels = unique(labels);

        if use_subplots
            n = numel(unique_labels);
            cols = ceil(sqrt(n));
            rows = ceil(n / cols);
            fig = figure('Color', 'w');

            for j = 1:n
                cluster_id = unique_labels(j);
                if cluster_id == -1, continue; end
                cluster_coords = coords(labels == cluster_id, :);
                if size(cluster_coords, 1) < min_cluster_size, continue; end

                subplot(rows, cols, j); hold on;
                try
                    shp = alphaShape(cluster_coords, alpha_value);
                    [tri, pts] = boundaryFacets(shp);
                    if isempty(tri), continue; end

                    if contains(mode, 'mesh')
                        p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
                        p.FaceAlpha = face_transparency;
                        p.EdgeColor = 'none';
                        p.FaceColor = [0.2+rand()*0.7, 0.5+rand()*0.4, 1.0];
                    end
                    if contains(mode, 'point')
                        scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 2, ...
                                 'filled', 'MarkerFaceAlpha', 0.1);
                    end
                    title(['Cluster ' num2str(cluster_id)]);
                    axis equal off;
                catch
                    continue;
                end
            end

            % Save full subplot view
            base_filename = fullfile(custom_output_dir, erase(mat_files(i).name, '.mat'));
%             savefig(fig, [base_filename '_subplots.fig']);
            saveas(fig, char(base_filename +'_subplots.png'));
            close(fig);
        else
            % Plot each cluster separately
            for j = 1:length(unique_labels)
                cluster_id = unique_labels(j);
                if cluster_id == -1, continue; end
                cluster_coords = coords(labels == cluster_id, :);
                if size(cluster_coords, 1) < min_cluster_size, continue; end

                try
                    shp = alphaShape(cluster_coords, alpha_value);
                    [tri, pts] = boundaryFacets(shp);
                    if isempty(tri), continue; end
                catch
                    continue;
                end

                fig = figure('Color','w'); hold on;
                if contains(mode, 'mesh')
                    p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
                    p.FaceAlpha = face_transparency; p.EdgeColor = 'none';
                    p.FaceColor = [0.2+rand()*0.7, 0.5+rand()*0.4, 1.0];
                end
                if contains(mode, 'point')
                    scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 2, ...
                             'filled', 'MarkerFaceAlpha', 0.1);
                end
                title(['Cluster ' num2str(cluster_id)]); xlabel('X'); ylabel('Y'); zlabel('Z');
                axis equal; view(3); camlight; lighting gouraud;

                base_filename = fullfile(custom_output_dir, ...
                    ['cluster_' num2str(cluster_id) '_' erase(mat_files(i).name, '.mat')]);

                % Save fig/png
%                 savefig(fig, (char(base_filename +'.fig'));
                saveas(fig, char(base_filename +'.png'));
                
                % Save rotating gif (optional)
                gif_filename = char(base_filename +'.gif');

%                 savefig(fig, char(base_filename + ".fig"));
%                 saveas(fig, char(base_filename + ".png"));
%                 gif_filename = char(base_filename + ".gif");

                for angle = 0:10:360
                    view(angle, 30); drawnow;
                    frame = getframe(fig); img = frame2im(frame);
                    [imind, cm] = rgb2ind(img, 256);
                    if angle == 0
                        imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.06);
                    else
                        imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.06);
                    end
                end
                close(fig);
            end
        end
    catch ME
        warning("❗ Could not process file %s due to error: %s", mat_path, ME.message);
    end
end

fprintf('\n✅ All processing and visualizations complete.\n');
end


%% 
% function clusterlabelCoordPlot_v2(data_dir, mode, alpha_value, min_cluster_size, face_transparency)
% % =========================================================================
% % clusterlabelCoordPlot_v2 - Visualize Labeled 3D Cluster Meshes from Files
% % =========================================================================
% % This function visualizes and saves 3D cluster meshes (via alphaShape) from
% % clustered coordinate data in .mat files, where each file contains [x,y,z,label].
% %
% % PARAMETERS:
% % -------------------------------------------------------------------------
% % data_dir            : Directory containing .mat files (each must have 4 cols)
% % mode                : 'mesh', 'point', or 'mesh+point'
% % alpha_value         : Alpha parameter for alphaShape (higher → smoother)
% % min_cluster_size    : Minimum points to form a mesh
% % face_transparency   : Transparency of mesh surface (0=clear, 1=solid)
% % =========================================================================
% 
% if nargin < 5, face_transparency = 0.6; end
% % if nargin < 4, min_cluster_size = 10; end
% if nargin < 4, min_cluster_size = 4; end
% if nargin < 3, alpha_value = 5; end
% if nargin < 2, mode = 'mesh'; end
% if nargin < 1 || isempty(data_dir)
%     error('Please provide a directory containing .mat files.');
% end
% 
% mat_files = dir(fullfile(data_dir, '*.mat'));
% 
% for i = 1:length(mat_files)
%     mat_path = fullfile(data_dir, mat_files(i).name);
%     fprintf("\nProcessing: %s\n", mat_path);
% 
%     try
%         % Load data directly assuming the variable is a matrix [x y z label]
%         loaded_struct = load(mat_path);
%         fieldNames = fieldnames(loaded_struct);  % fieldNames --> {'coords_mask_label'}
%         mat_data = loaded_struct.(fieldNames{1});  % First variable in file
% 
%         if size(mat_data, 2) ~= 4
%             warning("Skipped %s: Expected 4 columns (x,y,z,label), found %d.", mat_files(i).name, size(mat_data, 2));
%             continue;
%         end
% 
%         coords = mat_data(:, 1:3);
%         labels = mat_data(:, 4);
%         unique_labels = unique(labels);
% 
%         for j = 1:length(unique_labels)
%             cluster_id = unique_labels(j);
%             if cluster_id == -1, continue; end
%             cluster_coords = coords(labels == cluster_id, :);
%             if size(cluster_coords, 1) < min_cluster_size, continue; end
%             try
%                 cluster_coords = full(double(cluster_coords));
%                 shp = alphaShape(cluster_coords, alpha_value);
%                 [tri, pts] = boundaryFacets(shp);
%                 if isempty(tri), continue; end
%             catch
%                 continue;
%             end
% 
%             fig = figure('Color','w'); hold on;
%             if contains(mode, 'mesh')
%                 p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
%                 p.FaceAlpha = face_transparency; p.EdgeColor = 'none';
%                 p.FaceColor = [0.2+rand()*0.7, 0.5+rand()*0.4, 1.0];
%             end
%             if contains(mode, 'point')
%                 scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 2, 'filled', 'MarkerFaceAlpha', 0.1);
%             end
%             title(['Cluster ' num2str(cluster_id)]); xlabel('X'); ylabel('Y'); zlabel('Z');
%             axis equal; view(3); camlight; lighting gouraud;
% 
%             result_dir = fileparts(mat_path);
%             base_filename = fullfile(result_dir, ['cluster_' num2str(cluster_id) '_' erase(mat_files(i).name, '.mat')]);
% %             savefig(fig, char(base_filename + ".fig"));
%                 saveas(fig, char(base_filename + ".png"));
%                 gif_filename = char(base_filename + ".gif");
% 
%             % Create animated GIF
% %             gif_filename = [base_filename '.gif'];
%             for angle = 0:5:360
%                 view(angle, 30); drawnow;
%                 frame = getframe(fig); img = frame2im(frame);
%                 [imind, cm] = rgb2ind(img, 256);
%                 if angle == 0
%                     imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
%                 else
%                     imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
%                 end
%             end
%             close(fig);
%         end
%     catch ME
%         warning("Could not process file %s due to error: %s", mat_path, ME.message);
%     end
% end
% 
% fprintf('\n✅ All cluster visualizations are complete.\n');
% end






















% % -------------------------------------------------------------------------------------------------%%

% function clusterlabelCoordPlot_v2(data_dir, mode, alpha_value, min_cluster_size, face_transparency)
% % =========================================================================
% % 📊 clusterlabelCoordPlot_v2 - Visualize Labeled 3D Cluster Meshes from Files
% % =========================================================================
% % This function visualizes and saves 3D cluster meshes (via alphaShape) from
% % clustered coordinate data in .mat files, where each file contains [x,y,z,label].
% %
% % 🔧 PARAMETERS:
% % -------------------------------------------------------------------------
% % data_dir            : Directory containing .mat files (each must have 4 cols)
% % mode                : 'mesh', 'point', or 'mesh+point'
% % alpha_value         : Alpha parameter for alphaShape (higher → smoother)
% % min_cluster_size    : Minimum points to form a mesh
% % face_transparency   : Transparency of mesh surface (0=clear, 1=solid)
% % =========================================================================
% 
% if nargin < 5, face_transparency = 0.6; end
% if nargin < 4, min_cluster_size = 10; end
% if nargin < 3, alpha_value = 5; end
% if nargin < 2, mode = 'mesh'; end
% if nargin < 1 || isempty(data_dir)
%     error('Please provide a directory containing .mat files.');
% end
% 
% mat_files = dir(fullfile(data_dir, '*.mat'));
% 
% for i = 1:length(mat_files)
%     mat_path = fullfile(data_dir, mat_files(i).name);
%     fprintf("\nProcessing: %s\n", mat_path);
% 
%     try
%         alldata = load(mat_path);
%         fieldNames = fieldnames(alldata);
%         datavalues = alldata.(fieldNames{1});
% 
%         coords = datavalues(:,1:3);
%         labels = datavalues(:,4)';
%         unique_labels = unique(labels);
% 
%         for j = 1:length(unique_labels)
%             cluster_id = unique_labels(j);
%             if cluster_id == -1, continue; end
%             cluster_coords = coords(labels == cluster_id, :);
%             if size(cluster_coords,1) < min_cluster_size, continue; end
%             try
%                 cluster_coords = full(double(cluster_coords));
%                 shp = alphaShape(cluster_coords, alpha_value);
%                 [tri, pts] = boundaryFacets(shp);
%                 if isempty(tri), continue; end
%             catch
%                 continue;
%             end
% 
%             fig = figure('Color','w'); hold on;
%             if contains(mode, 'mesh')
%                 p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
%                 p.FaceAlpha = face_transparency; p.EdgeColor = 'none';
%                 p.FaceColor = [0.2+rand()*0.7, 0.5+rand()*0.4, 1.0];
%             end
%             if contains(mode, 'point')
%                 scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 2, 'filled', 'MarkerFaceAlpha', 0.1);
%             end
%             title(['Cluster ' num2str(cluster_id)]); xlabel('X'); ylabel('Y'); zlabel('Z');
%             axis equal; view(3); camlight; lighting gouraud;
% 
%             result_dir = fileparts(mat_path);
%             base_filename = fullfile(result_dir, ['cluster_' num2str(cluster_id) '_' erase(mat_files(i).name, '.mat')]);
% %             savefig(fig, [base_filename '.fig']);
% %             saveas(fig, [base_filename '.png']);
% 
%             saveas(fig, char(base_filename + ".png"));
%             gif_filename = char(base_filename + ".gif");
% 
%             % Create animated GIF
% %             gif_filename = [base_filename '.gif'];
%             for angle = 0:5:360
%                 view(angle, 30); drawnow;
%                 frame = getframe(fig); img = frame2im(frame);
%                 [imind, cm] = rgb2ind(img, 256);
%                 if angle == 0
%                     imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
%                 else
%                     imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
%                 end
%             end
%             close(fig);
%         end
%     catch ME
%         warning("Could not process file %s due to error: %s", mat_path, ME.message);
%     end
% end
% 
% fprintf('\nAll processing and visualizations complete.\n');
% end
