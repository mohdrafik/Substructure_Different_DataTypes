function visualize3DVolume(path2save, datamat, plotname, normalizedvolData, saveGif)
    % **Load the volume data**
    vol = datamat;

    % **Normalization Option**
    if normalizedvolData == 1
        fprintf(" Normalizing volume data...\n");
        if ~isa(vol, 'double')
            vol = double(vol);
        end
        vol = (vol - min(vol(:))) / (max(vol(:)) - min(vol(:)));
    else
        fprintf(" No normalization applied.\n");
    end

    % **Create 3D Volume Visualization**
    hvol = volshow(vol);
    viewer = hvol.Parent;
    hFig = viewer.Parent;
    sz = size(vol);
    center = sz / 2 + 0.5;
    viewer.CameraTarget = center;
    zoom 'on'

    % **Colormap & Transparency Settings**
    cmap = hot(256);                 % Brighter Inner Core & Transparent Outer
    alpha = linspace(0, 1, 256).^0.5; % Inner bright, outer soft-fade
    alpha(1:20) = 0;                  % Transparent outer layers

    % **Alternative Colormap Options (commented)**
    
    % Completely Transparent Outer Volume - Great for seeing internal structures
    % cmap = gray(256); alpha = linspace(0, 1, 256).^1.5; alpha(1:50) = 0; 

    % High Contrast Jet Colormap - Better for scientific data (sharp contrast)
    % cmap = jet(256); alpha = linspace(0, 1, 256).^1.2; alpha(1:10) = 0;  

    % **Apply Colormap & Transparency**
    hvol.Colormap = cmap;  
    hvol.Alphamap = alpha;

    % **Camera Animation Settings**
    numberOfFrames = 60;
    vec = linspace(0, 2*pi, numberOfFrames)';
    dist = sqrt(sz(1)^2 + sz(2)^2 + sz(3)^2);
    myPosition = center + ([cos(vec) sin(vec) ones(size(vec))] * dist);

    % **Prepare GIF Saving Settings (if enabled)**
    if saveGif
        filename = fullfile(path2save, plotname + ".gif");
        delayTime = 0.15;  % Adjust delay for better visualization
    end

    % **Loop to Generate GIF**
    for idx = 1:length(vec)
        viewer.CameraPosition = myPosition(idx, :);
        I = getframe(hFig);
        [indI, cm] = rgb2ind(I.cdata, 256);
        
        % **Write the frame to the GIF file (if enabled)**
        if saveGif
            if idx == 1
                % Skip first frame to avoid blank display
            elseif idx == 2
                imwrite(indI, cm, filename, "gif", "Loopcount", inf, "DelayTime", delayTime);
            else
                imwrite(indI, cm, filename, "gif", "WriteMode", "append", "DelayTime", delayTime);
            end
        end
    end

    % **Print Message (if GIF was saved)**
    if saveGif
        fprintf(" Saved GIF: %s\n", filename);
    end
end
