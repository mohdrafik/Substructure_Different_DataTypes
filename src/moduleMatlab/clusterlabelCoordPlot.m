function clusterlabelCoordPlot(choose, root_dir, mode, alpha_value, min_cluster_size, face_transparency)
% =========================================================================
% 📊 clusterMeshVisualizer - Cluster Mesh and Visualization Function
% =========================================================================
% This function visualizes and saves 3D cluster meshes (via alphaShape) from
% clustered coordinate data in .mat files.
%
% 🔧 PARAMETERS:
% -------------------------------------------------------------------------
% choose              : 0 → use individual cluster_* directories
%                       1 → use final merged 'kmeans_coords_labels.mat' format
% root_dir            : Path to the root directory (if empty, default is used)
% mode                : 'mesh', 'point', or 'mesh+point'
% alpha_value         : Alpha parameter for alphaShape (higher → smoother)
% min_cluster_size    : Minimum points to form a mesh
% face_transparency   : Transparency of mesh surface (0=clear, 1=solid)
% =========================================================================

if nargin < 6, face_transparency = 0.6; end
if nargin < 5, min_cluster_size = 10; end
if nargin < 4, alpha_value = 5; end
if nargin < 3, mode = 'mesh'; end
if nargin < 2 || isempty(root_dir)
    root_dir = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\hybrid_Kdbcluster';
    fprintf("Default root_dir selected: %s\n", root_dir);
end

switch choose
    case 0
        clusterFolders = dir(fullfile(root_dir, 'cluster_*'));
        for k = 1:length(clusterFolders)
            folderName = clusterFolders(k).name;
            fullPath = fullfile(root_dir, folderName);
            tokens = regexp(folderName, 'cluster_(\d+)', 'tokens');
            if isempty(tokens), continue; end
            clusterID = tokens{1}{1};
            clusterMatPath = fullfile(fullPath, sprintf('cluster_%s.mat', clusterID));
            coordsMatPath  = fullfile(fullPath, sprintf('coords_%s.mat', clusterID));

            if isfile(clusterMatPath), labels = load(clusterMatPath); else, continue; end
            if isfile(coordsMatPath), coords = load(coordsMatPath);  else, continue; end
            cluster_coords = coords.coords;

            if size(cluster_coords,1) < min_cluster_size, continue; end
            try
                cluster_coords = full(double(cluster_coords));
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
                scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 2, 'filled', 'MarkerFaceAlpha', 0.1);
            end
            title(['Cluster ' clusterID]); xlabel('X'); ylabel('Y'); zlabel('Z');
            axis equal; view(3); camlight; lighting gouraud;
            base_filename = fullfile(fullPath, ['cluster_' clusterID]);
            savefig(fig, char(base_filename + ".fig"));
            saveas(fig, char(base_filename + ".png"));
            gif_filename = char(base_filename + ".gif");
            for angle = 0:5:360
                view(angle, 30); drawnow;
                frame = getframe(fig); img = frame2im(frame);
                [imind, cm] = rgb2ind(img, 256);
                if angle == 0
                    imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
                else
                    imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
                end
            end
            close(fig);
        end

    case 1
        target_dirs = dir(fullfile(root_dir, 'kmIntensity*'));
        for i = 1:length(target_dirs)
            folder_path = fullfile(root_dir, target_dirs(i).name);
            mat_file = fullfile(folder_path, 'kmeans_coords_labels.mat');
            if ~exist(mat_file, 'file'), continue; end
            fprintf(" Processing: %s\n", mat_file);
            alldata = load(mat_file);
            fieldNames = fieldnames(alldata);
            labels = alldata.(fieldNames{1})(:,4)';
            coords = alldata.(fieldNames{1})(:,1:3);
            unique_labels = unique(labels);
            for j = 1:length(unique_labels)
                cluster_id = unique_labels(j);
                if cluster_id == -1, continue; end
                cluster_coords = coords(labels == cluster_id, :);
                if size(cluster_coords,1) < min_cluster_size, continue; end
                try
                    cluster_coords = full(double(cluster_coords));
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
                    scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 2, 'filled', 'MarkerFaceAlpha', 0.1);
                end
                title(['Cluster ' num2str(cluster_id)]); xlabel('X'); ylabel('Y'); zlabel('Z');
                axis equal; view(3); camlight; lighting gouraud;
                result_dir = fileparts(mat_file);
                base_filename = fullfile(result_dir, ['cluster_' num2str(cluster_id)]);
                savefig(fig, char(base_filename + ".fig"));
                saveas(fig, char(base_filename + ".png"));
                gif_filename = char(base_filename + ".gif");
                for angle = 0:5:360
                    view(angle, 30); drawnow;
                    frame = getframe(fig); img = frame2im(frame);
                    [imind, cm] = rgb2ind(img, 256);
                    if angle == 0
                        imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
                    else
                        imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
                    end
                end
                close(fig);
            end
        end

    otherwise
        disp(' Invalid input. Choose 0 or 1.');
end

fprintf('\n All processing and visualizations are complete.\n');
end



%% <------------------------------ this below code was working and showing mesh and point 3d interactive plots ----------------------->

% clc;
% clear ;
% close all;
% 
% choose = input("enter the value 0 for all dbscan plot for individual clusters or 1 for extract labels and coordinates from final combined data:" );
% % Set the root directory where all cluster folders exist
% 
% switch choose
%     case 0
% 
%         rootDir = 'E:\Projects\substructure_3d_data\Substructure_Different_DataTypes\results\hybrid_Kdbcluster\Tomogramma_BuddingYeastCell';  % 🔁 CHANGE this to your real path
% 
%         % Get all cluster folders (starting with 'cluster_')
%         clusterFolders = dir(fullfile(rootDir, 'cluster_*'));
% 
%         % Loop through each cluster folder
%         for k = 1:length(clusterFolders)
%             folderName = clusterFolders(k).name;
%             fullPath = fullfile(rootDir, folderName);
% 
%             % Extract cluster number from folder name
%             %     tokens = regexp(folderName, 'cluster_(\d+)', 'tokens');
%             % Extract cluster number from folder name
%             tokens = regexp(folderName, 'cluster_(\d+)', 'tokens');
%             if isempty(tokens)
%                 warning('Skipping unmatched folder: %s\n', folderName);
%                 continue;  % Skip to next iteration
%             end
%             clusterID = tokens{1}{1};  % Now safe to access
% 
% 
% 
%             % Build full paths to .mat files
%             clusterMatPath = fullfile(fullPath, sprintf('cluster_%s.mat', clusterID));
%             coordsMatPath = fullfile(fullPath, sprintf('coords_%s.mat', clusterID));
% 
%             % Load .mat files
%             if isfile(clusterMatPath)
%                 labels = load(clusterMatPath);
%                 fprintf('Loaded %s\n', clusterMatPath);
%             else
%                 warning('Missing: %s\n', clusterMatPath);
%             end
% 
%             if isfile(coordsMatPath)
%                 coords = load(coordsMatPath);
%                 fprintf('Loaded %s\n', coordsMatPath);
%             else
%                 warning('Missing: %s\n', coordsMatPath);
%             end
% 
%             % here is now process clusterData and coordsData here...
% 
% 
%             % === SETTINGS ===
% 
%             % mode = 'mesh+point';   % Options: 'mesh', 'point', 'mesh+point'
%             mode = 'mesh';
%             alpha_value = 5;       % Alpha parameter for alphaShape
%             min_cluster_size = 10; % Minimum number of points to create mesh
%             face_transparency = 0.6;
% 
%             cluster_coords = coords.coords ;
%             % cluster_id = k-1; cluster_id = str2double(tokens{1}{1});
%             cluster_id = clusterID;
% 
%             if size(cluster_coords, 1) < min_cluster_size
%                 fprintf("  Cluster %d skipped (too small)\n", cluster_id);
%                 continue;
%             end
% 
%             % === Generate alpha shape ===
%             try
%                 cluster_coords = full(double(cluster_coords)); %  forcefully converted the matrix to double, full, and real before alphaShape if not.
%                 shp = alphaShape(cluster_coords, alpha_value);
%                 [tri, pts] = boundaryFacets(shp);
% 
%                 if isempty(tri)
%                     fprintf("  Cluster %d: Could not form valid mesh (empty facets)\n", cluster_id);
%                     continue;
%                 end
%             catch ME
%                 fprintf("  Cluster %d: Alpha shape error - %s\n", cluster_id, ME.message);
%                 continue;
%             end
% 
%             % === Plotting ===
%             figure('Color','w');
%             hold on;
% 
%             % Mesh mode
%             if strcmp(mode, 'mesh') || strcmp(mode, 'mesh+point')
%                 p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
%                 p.FaceAlpha = face_transparency;
%                 p.EdgeColor = 'none';
%                 p.FaceColor = [0.2+rand(1)*0.7, 0.5+rand(1)*0.4, 1.0];  % Nice color gradient
%             end
% 
%             % Point mode
%             if strcmp(mode, 'point') || strcmp(mode, 'mesh+point')
%                 scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), ...
%                     2, 'filled', 'MarkerFaceAlpha', 0.1);
%             end
% 
%             title(['Cluster ' num2str(cluster_id)]);
%             xlabel('X'); ylabel('Y'); zlabel('Z');
%             axis equal;
%             grid on;
%             view(3); camlight; lighting gouraud;
%         end
% 
% 
% 
%         %% this below code was working fine, above i just looped for all individual cluster finally after km-dbScan .............
% 
%     case 1
% 
% 
%         % === SETTINGS === mode = 'mesh+point';   % Options: 'mesh', 'point',
%         % 'mesh+point'
%         mode = 'mesh';
%         alpha_value = 5;       % Alpha parameter for alphaShape
%         min_cluster_size = 10;  % Minimum number of points to create mesh
%         face_transparency = 0.8;
% 
%         user_input = input("Enter root directory path or press Enter to use default: ", 's');
%         if isempty(user_input)
%             root_dir = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\hybrid_Kdbcluster';
%             fprintf("Default root_dir selected: %s\n", root_dir);
%         else
%             root_dir = user_input;
%             fprintf("User root_dir selected: %s\n", root_dir);
%         end
% 
%         %         root_dir = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\hybrid_Kdbcluster';
% 
%         target_dirs = dir(fullfile(root_dir, 'kmIntensity*'));
% 
%         for i = 1:length(target_dirs)
%             folder_path = fullfile(root_dir, target_dirs(i).name);
%             mat_file = fullfile(folder_path, 'kmeans_coords_labels.mat');
% 
%             if exist(mat_file, 'file')
%                 fprintf(" Processing: %s\n", mat_file);
% 
% 
%                 %         filepath = input('enter the complete filepath: enclose your path with quotes as a string');
%                 %         disp(filepath)
% 
%                 filepath = mat_file;
%                 alldata = load(filepath);
% 
%                 fieldNames = fieldnames(alldata);
%                 % Access the first field (assuming only one data variable exists)
%                 labels = alldata.(fieldNames{1})(:,4);
%                 % labels = data.kmeans_coords_labels(:,4);
%                 labels = labels';
% 
%                 coords = alldata.(fieldNames{1})(:,1:3);
% 
% 
%                 unique_labels = unique(labels);
% 
%                 % === Loop through each cluster ===
%                 for j = 1:length(unique_labels)
%                     cluster_id = unique_labels(j);
% 
%                     if cluster_id == -1
%                         continue; % Skip noise
%                     end
% 
%                     cluster_coords = coords(labels == cluster_id, :);
% 
%                     if size(cluster_coords, 1) < min_cluster_size
%                         fprintf("  Cluster %d skipped (too small)\n", cluster_id);
%                         continue;
%                     end
% 
%                     % === Generate alpha shape ===
%                     try
%                         cluster_coords = full(double(cluster_coords)); %  forcefully converted the matrix to double, full, and real before alphaShape if not.
%                         shp = alphaShape(cluster_coords, alpha_value);
%                         [tri, pts] = boundaryFacets(shp);
% 
%                         if isempty(tri)
%                             fprintf("  Cluster %d: Could not form valid mesh (empty facets)\n", cluster_id);
%                             continue;
%                         end
%                     catch ME
%                         fprintf("  Cluster %d: Alpha shape error - %s\n", cluster_id, ME.message);
%                         continue;
%                     end
% 
%                     % === Plotting ===
% 
%                     fig = figure('Color','w');
%                     hold on;
% 
%                     % Mesh mode
%                     if strcmp(mode, 'mesh') || strcmp(mode, 'mesh+point')
%                         p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
%                         p.FaceAlpha = face_transparency;
%                         p.EdgeColor = 'none';
%                         p.FaceColor = [0.2+rand(1)*0.7, 0.5+rand(1)*0.4, 1.0];  % Nice color gradient
%                     end
% 
%                     % Point mode
%                     if strcmp(mode, 'point') || strcmp(mode, 'mesh+point')
%                         scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), ...
%                             2, 'filled', 'MarkerFaceAlpha', 0.1);
%                     end
% 
%                     title(['Cluster ' num2str(cluster_id)]);
%                     xlabel('X'); ylabel('Y'); zlabel('Z');
%                     axis equal;
%                     %             grid on;
%                     view(3); camlight; lighting gouraud;
% 
%                     % === Saving ===
% 
%                     %             result_dir = "C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\hybrid_Kdbcluster\kmIntensitytomo_grafene_48h";
% 
%                     result_dir = fileparts(filepath);
%                     base_filename = fullfile(result_dir, ['cluster_' num2str(cluster_id)]);
% 
%                     % Save as .fig
%                     % savefig(fig, [base_filename,'.fig']);
%                     savefig(fig, char(base_filename + ".fig"));    %
%                     % Save as .png
% 
%                     % saveas(fig, [base_filename,'.png']);
%                     saveas(fig, char(base_filename + ".png"));      %
% 
% 
%                     % Optional: Save rotating GIF
%                     %             gif_filename = [base_filename,'.gif'];
%                     gif_filename = char(base_filename + ".gif");
% 
%                     for angle = 0:5:360
%                         view(angle, 30);
%                         drawnow;
%                         frame = getframe(fig);
%                         img = frame2im(frame);
%                         [imind, cm] = rgb2ind(img, 256);
%                         if angle == 0
%                             imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.05);
%                         else
%                             imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.05);
%                         end
%                     end
%                     close(fig);
%                 end
%                 fprintf('All plotting is finished with files are saved at : --> %s \n',result_dir);
%             end
%         end
% 
%     otherwise
%         disp('other value when you have not given appropriate choose value:')
% end
% 
% 




%% <------------------------------this below code was working and showing mesh and point 3d interactive plots ----------------------->

% % === Load cluster labels and coordinates ===

% labels =
% load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\cluster_labels.mat');
% coords =
% load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\voxel_coords.mat');
%
% labels = labels.data; coords = coords.data;
%
% unique_labels = unique(labels);
%
% % === Loop through clusters === for i = 1:length(unique_labels)
%     cluster_id = unique_labels(i);
%
%     % Skip DBSCAN noise if cluster_id == -1
%         continue;
%     end
%
%     % Extract points of the current cluster cluster_coords =
%     coords(labels == cluster_id, :);
%
%     % Skip very small clusters if size(cluster_coords, 1) < 6  % earlier
%     30,
%         continue;
%     end
%
%     % === Create alpha shape for mesh === shp =
%     alphaShape(cluster_coords, 5);  % adjust alpha as needed [tri, pts] =
%     boundaryFacets(shp);     % get triangular surface
%
%     % === Plot both points and mesh === figure('Color', 'w'); hold on;
%
%     % Plot surface mesh trisurf(tri, pts(:,1), pts(:,2), pts(:,3), ...
%         'FaceColor', [0.1 + rand*0.8, 0.4, 0.9], ... 'FaceAlpha', 0.3,
%         'EdgeColor', 'none');
%
%     % Plot original points scatter3(cluster_coords(:,1),
%     cluster_coords(:,2), cluster_coords(:,3), 2, 'filled');
%
%     title(['Cluster ' num2str(cluster_id)]); xlabel('X'); ylabel('Y');
%     zlabel('Z'); axis equal; grid on; view(3); camlight; lighting
%     gouraud;
%
% end


%% <------------------------------this below code is working and showing point 3d plots only SIMPLE PLOT JSUT TO CHECK ----------------------->

% labels =
% load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\cluster_labels.mat')
% ;     % shape: [N, 1] coords =
% load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\voxel_coords.mat');
%
% labels = labels.data; coords = coords.data;
%
% unique_labels = unique(labels);
%
% for i = 1:length(unique_labels)
%     cluster_id = unique_labels(i);
%
%     % Skip noise label (usually -1 in DBSCAN)
%
%     if cluster_id == -1
%         continue;
%     end
%
%     % Get coordinates for this cluster cluster_coords = coords(labels ==
%     cluster_id, :);
%
%     % Plot figure; scatter3(cluster_coords(:,1), cluster_coords(:,2),
%     cluster_coords(:,3), 1, 'filled');
%
%     title(['Cluster ' num2str(cluster_id)]); xlabel('X'); ylabel('Y');
%     zlabel('Z'); grid on; axis equal;

% end


