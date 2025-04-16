% Load the .npy files (requires the 'readNPY' function from npy-matlab)


clc;
clear all;
close all;

% === SETTINGS ===
% mode = 'mesh+point';   % Options: 'mesh', 'point', 'mesh+point'
mode = 'mesh';
alpha_value = 5;       % Alpha parameter for alphaShape
min_cluster_size = 10; % Minimum number of points to create mesh
face_transparency = 0.6;

% === Load Data ===
labels = load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\cluster_labels.mat');
coords = load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\voxel_coords.mat');

labels = labels.data;
coords = coords.data;

unique_labels = unique(labels);

% === Loop through each cluster ===
for i = 1:length(unique_labels)
    cluster_id = unique_labels(i);

    if cluster_id == -1
        continue; % Skip noise
    end

    cluster_coords = coords(labels == cluster_id, :);

    if size(cluster_coords, 1) < min_cluster_size
        fprintf("⚠️ Cluster %d skipped (too small)\n", cluster_id);
        continue;
    end

    % === Generate alpha shape ===
    try
        shp = alphaShape(cluster_coords, alpha_value);
        [tri, pts] = boundaryFacets(shp);

        if isempty(tri)
            fprintf("⚠️ Cluster %d: Could not form valid mesh (empty facets)\n", cluster_id);
            continue;
        end
    catch ME
        fprintf("⚠️ Cluster %d: Alpha shape error - %s\n", cluster_id, ME.message);
        continue;
    end

    % === Plotting ===
    figure('Color','w');
    hold on;

    % Mesh mode
    if strcmp(mode, 'mesh') || strcmp(mode, 'mesh+point')
        p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3));
        p.FaceAlpha = face_transparency;
        p.EdgeColor = 'none';
        p.FaceColor = [0.2+rand(1)*0.7, 0.5+rand(1)*0.4, 1.0];  % Nice color gradient
    end

    % Point mode
    if strcmp(mode, 'point') || strcmp(mode, 'mesh+point')
        scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), ...
            2, 'filled', 'MarkerFaceAlpha', 0.1);
    end

    title(['Cluster ' num2str(cluster_id)]);
    xlabel('X'); ylabel('Y'); zlabel('Z');
    axis equal;
    grid on;
    view(3); camlight; lighting gouraud;
end



%% <------------------------------this below code was working and showing mesh and point 3d interactive plots ----------------------->  

% % === Load cluster labels and coordinates ===

% labels = load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\cluster_labels.mat');
% coords = load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\voxel_coords.mat');
% 
% labels = labels.data;
% coords = coords.data;
% 
% unique_labels = unique(labels);
% 
% % === Loop through clusters ===
% for i = 1:length(unique_labels)
%     cluster_id = unique_labels(i);
% 
%     % Skip DBSCAN noise
%     if cluster_id == -1
%         continue;
%     end
% 
%     % Extract points of the current cluster
%     cluster_coords = coords(labels == cluster_id, :);
% 
%     % Skip very small clusters
%     if size(cluster_coords, 1) < 6  % earlier 30, 
%         continue;
%     end
% 
%     % === Create alpha shape for mesh ===
%     shp = alphaShape(cluster_coords, 5);  % adjust alpha as needed
%     [tri, pts] = boundaryFacets(shp);     % get triangular surface
% 
%     % === Plot both points and mesh ===
%     figure('Color', 'w');
%     hold on;
% 
%     % Plot surface mesh
%     trisurf(tri, pts(:,1), pts(:,2), pts(:,3), ...
%         'FaceColor', [0.1 + rand*0.8, 0.4, 0.9], ...
%         'FaceAlpha', 0.3, 'EdgeColor', 'none');
% 
%     % Plot original points
%     scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 2, 'filled');
% 
%     title(['Cluster ' num2str(cluster_id)]);
%     xlabel('X'); ylabel('Y'); zlabel('Z');
%     axis equal;
%     grid on;
%     view(3);
%     camlight; lighting gouraud;
% 
% end


%% <------------------------------this below code is working and showing point 3d plots only SIMPLE PLOT JSUT TO CHECK ----------------------->  
 
% labels = load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\cluster_labels.mat') ;     % shape: [N, 1]
% coords = load('C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\src\clustering_output\convertedmatfiles\voxel_coords.mat');
% 
% labels = labels.data;
% coords = coords.data;
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
%     % Get coordinates for this cluster
%     cluster_coords = coords(labels == cluster_id, :);
% 
%     % Plot
%     figure;
%     scatter3(cluster_coords(:,1), cluster_coords(:,2), cluster_coords(:,3), 1, 'filled');
% 
%     title(['Cluster ' num2str(cluster_id)]);
%     xlabel('X'); ylabel('Y'); 
%     zlabel('Z');
%     grid on; 
%     axis equal;

% end


