clc;
clear all;
close all;

% **Get user input for .MAT file directory**
datapathDotMatFiles = input("Enter the .mat file data path (or press Enter/'N' to use default path): ", "s");

if isempty(datapathDotMatFiles) || strcmpi(datapathDotMatFiles, 'N')
    data_dir = "C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\data\intermdata1\";
    fprintf("✅ Data path set to default: %s \n", data_dir);
else
    data_dir = datapathDotMatFiles;
    fprintf("✅ Data path chosen: %s\n", data_dir);
end

% **Define path to save plots**
path2save = "C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\plots\normalizeVolplot\";

% **Ensure output directory exists**
if ~exist(path2save, 'dir')
    mkdir(path2save);
end

% **Ask for Normalization & GIF Saving**
normalizedvolData = input("Do you want normalization of volData? (Enter 1 for YES, 0 for NO): ");
saveGif = input("Do you want to save GIF images? (Enter 1 for YES, 0 for NO): ");

% **List all .MAT files**
file_list = dir(fullfile(data_dir, '*.mat'));

% **Check if any .MAT files are present**
num_files = length(file_list);
if num_files == 0
    error("❌ No .MAT files found in the directory: %s", data_dir);
end

fprintf("\n🔍 Found %d .MAT files. Processing...\n", num_files);

% **Loop through each .MAT file**
for i = 1:num_files
    filename = file_list(i).name;
    fullpath = fullfile(data_dir, filename);

    fprintf("\n📂 <----- Processing File: %s (%d/%d) ----->\n", filename, i, num_files);

    try
        % **Load .MAT file**
        dataStruct = load(fullpath);
    catch
        fprintf("⚠ Skipping %s (Unreadable or Corrupted)\n", filename);
        continue;
    end

    fieldNames = fieldnames(dataStruct); % Get field names

    if isempty(fieldNames)
        fprintf("⚠ Warning: No valid fields found in %s. Skipping...\n", filename);
        continue; % Skip this file
    end

    % **Extract first field (Assuming first field contains required data)**
    volData = dataStruct.(fieldNames{1});

    % **Check if data is 3D**
    if ndims(volData) ~= 3
        fprintf("⚠ Warning: Skipping %s (Not a 3D dataset)\n", filename);
        continue;
    end

    % **Generate Filename for GIF**
    gifFilename = extractBefore(filename, '.mat');  % More reliable than erase()

    % **Call Visualization Function**
    visualize3DVolume(path2save, volData, gifFilename, normalizedvolData, saveGif);
end

fprintf("\n✅ All .MAT files processed successfully!\n");


