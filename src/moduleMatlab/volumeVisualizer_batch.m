% volumeVisualizer_batch
function volumeVisualizer_batch(data_dir, mode, alpha_value, intensity_threshold, face_transparency)
% =========================================================================
% 📊 volumeVisualizer_batch - 3D Volume Visualization (mesh/points)
% =========================================================================
% Visualizes 3D intensity volumes from .mat files containing a single 3D array.
% Automatically handles thresholding and saves PNG + GIF.
%
% 🔧 PARAMETERS:
%   data_dir : Directory of .mat files containing volume (3D array)
%   mode     : 'mesh', 'point', or 'mesh+point'
%   alpha_value : alphaShape smoothness (for mesh)
%   intensity_threshold : threshold for visible voxels (e.g., >0)
%   face_transparency: transparency for mesh faces (0–1)

% use case: 
% data_dir = 'C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\significantDigits_json10oct2025';
% volumeVisualizer_batch(data_dir, 'mesh+point', 5, 0.2, 0.6); 
%  OR -----------------> 
% volumeVisualizer_batch(data_dir);

% =========================================================================

if nargin < 5, face_transparency = 0.5; end
if nargin < 4, intensity_threshold = 0; end
if nargin < 3, alpha_value = 5; end
if nargin < 2, mode = 'mesh+point'; end
if nargin < 1, error('Please provide a data_dir'); end

output_dir = fullfile(data_dir, 'visualization_results_volume');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

mat_files = dir(fullfile(data_dir, '*.mat'));
fprintf("🔎 Found %d mat files.\n", length(mat_files));

for i = 1:length(mat_files)
    mat_path = fullfile(data_dir, mat_files(i).name);
    fprintf("\n📂 Processing: %s\n", mat_files(i).name);

    try
        % --- Load volume ---
        alldata = load(mat_path);
        fieldNames = fieldnames(alldata);
        volume = alldata.(fieldNames{1});

        if ndims(volume) ~= 3
            fprintf("⚠️ Skipping %s — not a 3D volume.\n", mat_files(i).name);
            continue;
        end

        % --- Extract voxel coordinates above threshold ---
        [x, y, z] = ind2sub(size(volume), find(volume > intensity_threshold));
        intensities = volume(volume > intensity_threshold);
        coords = [x(:), y(:), z(:)];

        fprintf("💡 %d voxels above threshold %.2f\n", size(coords,1), intensity_threshold);

        if isempty(coords)
            fprintf("⚠️ No visible voxels for %s\n", mat_files(i).name);
            continue;
        end

        % --- Plot ---
        fig = figure('Color','w'); hold on;
        plotVolume(coords, intensities, mode, alpha_value, face_transparency);

        title(erase(mat_files(i).name,'.mat'),'Interpreter','none');
        xlabel('X'); ylabel('Y'); zlabel('Z');
        axis equal; view(3); camlight; lighting gouraud;
        grid off; box on;

        % --- Save PNG ---
        base_filename = fullfile(output_dir, erase(mat_files(i).name,'.mat'));
        saveas(fig, [base_filename '.png']);

        % --- Save rotating GIF ---
        gif_filename = [base_filename '.gif'];
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

    catch ME
        fprintf("❌ Error processing %s: %s\n", mat_files(i).name, ME.message);
    end
end

fprintf("\n✅ All processing complete.\n");
end

% =========================================================================
% 🧩 Helper plotting function
% =========================================================================
function plotVolume(coords, intensities, mode, alpha_value, face_transparency)
if contains(mode, 'mesh')
    try
        shp = alphaShape(coords, alpha_value);
        [tri, pts] = boundaryFacets(shp);
        if ~isempty(tri)
            p = trisurf(tri, pts(:,1), pts(:,2), pts(:,3), ...
                        'FaceAlpha', face_transparency, ...
                        'EdgeColor', 'none', ...
                        'FaceColor', [0.2+rand()*0.7, 0.5+rand()*0.4, 1.0]);
        else
            fprintf("🚩 Empty mesh — fallback to point plot.\n");
        end
    catch
        fprintf("🚩 alphaShape failed — fallback to points.\n");
    end
end

if contains(mode, 'point') || ~exist('p','var')
    scatter3(coords(:,1), coords(:,2), coords(:,3), 3, intensities, 'filled', ...
             'MarkerFaceAlpha', 0.4);
    colormap('turbo');
end
end
