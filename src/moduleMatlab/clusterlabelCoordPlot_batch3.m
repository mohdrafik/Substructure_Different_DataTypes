function clusterlabelCoordPlot_batch3(data_dir, mode, alpha_value, min_cluster_size, face_transparency, use_subplots)
% =========================================================================
% 📊 clusterlabelCoordPlot_v2 - Robust version with fallback plotting
% =========================================================================
% Visualizes and saves 3D cluster meshes or points from .mat files.
% Works even if alphaShape fails, always saves PNG + GIF.
%
% 🔧 PARAMETERS:
%   data_dir : Directory of .mat files containing [x y z label]
%   mode     : 'mesh', 'point', or 'mesh+point'
%   alpha_value : alphaShape smoothness
%   min_cluster_size : minimum points to visualize
%   face_transparency: 0 to 1
%   use_subplots : true/false (all clusters in one fig or separate figs)
% =========================================================================

if nargin < 6, use_subplots = false; end
if nargin < 5, face_transparency = 0.6; end
if nargin < 4, min_cluster_size = 10; end
if nargin < 3, alpha_value = 5; end
if nargin < 2, mode = 'mesh+point'; end
if nargin < 1, error('Please provide a data_dir'); end

custom_output_dir = fullfile(data_dir, 'visualization_results_mesh');
if ~exist(custom_output_dir, 'dir')
    mkdir(custom_output_dir);
end

mat_files = dir(fullfile(data_dir, '*.mat'));
fprintf("🔎 Found %d mat files.\n", length(mat_files));

for i = 1:length(mat_files)
    mat_path = fullfile(data_dir, mat_files(i).name);
    fprintf("\n📂 Processing: %s\n", mat_files(i).name);

    try
        alldata = load(mat_path);
        fieldNames = fieldnames(alldata);
        data_matrix = alldata.(fieldNames{1});

        coords = data_matrix(:,1:3);
        labels = data_matrix(:,4);
        unique_labels = unique(labels);

        fprintf("Unique labels: %s\n", mat2str(unique_labels'));

        if use_subplots
            n = numel(unique_labels);
            cols = ceil(sqrt(n));
            rows = ceil(n/cols);
            fig = figure('Color', 'k');

            for j = 1:n
                cluster_id = unique_labels(j);
                cluster_coords = coords(labels == cluster_id, :);
                if size(cluster_coords,1) < min_cluster_size
                    fprintf("⚠️ Skipping small cluster %d (%d pts)\n", cluster_id, size(cluster_coords,1));
                    continue;
                end
                subplot(rows, cols, j); hold on;
                plotCluster(cluster_coords, cluster_id, mode, alpha_value, face_transparency);
                axis equal off;
            end

            saveas(fig, fullfile(custom_output_dir, [erase(mat_files(i).name,'.mat') '_subplots.png']));
            close(fig);

        else
            for j = 1:length(unique_labels)
                cluster_id = unique_labels(j);
                cluster_coords = coords(labels == cluster_id, :);

                if size(cluster_coords,1) < min_cluster_size
                    fprintf("⚠️ Skipping cluster %d (%d pts)\n", cluster_id, size(cluster_coords,1));
                    continue;
                end

                fig = figure('Color','w'); hold on;
                plotCluster(cluster_coords, cluster_id, mode, alpha_value, face_transparency);
                title(['Cluster ' num2str(cluster_id)]);
                xlabel('X'); ylabel('Y'); zlabel('Z');
                axis equal; view(3); camlight; lighting gouraud;

                base_filename = fullfile(custom_output_dir, ...
                    ['cluster_' num2str(cluster_id) '_' erase(mat_files(i).name,'.mat')]);

                % Save PNG
                saveas(fig, [base_filename '.png']);
                
                % Save GIF
                gif_filename = [base_filename '.gif'];
                for angle = 0:10:360
                    view(angle,30); drawnow;
                    frame = getframe(fig); img = frame2im(frame);
                    [imind,cm] = rgb2ind(img,256);
                    if angle==0
                        imwrite(imind,cm,gif_filename,'gif','Loopcount',inf,'DelayTime',0.06);
                    else
                        imwrite(imind,cm,gif_filename,'gif','WriteMode','append','DelayTime',0.06);
                    end
                end
                close(fig);
            end
        end

    catch ME
        fprintf("❌ Error processing %s: %s\n", mat_files(i).name, ME.message);
    end
end
fprintf("\n✅ All processing complete.\n");
end

% Helper plotting function
function plotCluster(cluster_coords, cluster_id, mode, alpha_value, face_transparency)
if contains(mode, 'mesh')
    try
        shp = alphaShape(cluster_coords, alpha_value);
        [tri, pts] = boundaryFacets(shp);
        if ~isempty(tri)
            p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
            p.FaceAlpha = face_transparency;
            p.EdgeColor = 'none';
            p.FaceColor = [0.2+rand()*0.7, 0.5+rand()*0.4, 1.0];
        else
            fprintf("🚩 Mesh empty for cluster %d — fallback to points\n", cluster_id);
        end
    catch
        fprintf("🚩 alphaShape failed for cluster %d — fallback to points\n", cluster_id);
    end
end
if contains(mode,'point') || isempty(tri)
    scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 4, ...
             'filled', 'MarkerFaceAlpha', 0.3);
end
end
