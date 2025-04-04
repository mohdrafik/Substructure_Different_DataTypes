clc
clear all
close all;
folderPath = "C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\data\normalized_matData\";
matFiles = dir(fullfile(folderPath, '*.mat'));




for k = 1:length(matFiles)
    fileName = matFiles(k).name;
    disp(fileName);
    fullPath = fullfile(folderPath, fileName);
    
    % Load data
    loaded = load(fullPath);
    varNames = fieldnames(loaded);
    data = loaded.(varNames{1}); % assuming single variable
    
    % Reshape if needed
    data = data(:);
    gmm1 = fitgmdist(data,k,'RegularizationValue',1e-5);
    means= gmm1.mu;
    weights = gmm1.ComponentProportion;
    sigmas = sqrt(squeeze(gmm1.Sigma));
    [~, idx_max] = max(weights);
    most_repetitive_value = means(idx_max);

    sigma_peak = sigmas(idx_max);

    % Histogram
figure;
histogram(data, 100, 'Normalization', 'pdf'); hold on;

% Overlay GMM curve
x_vals = linspace(min(data), max(data), 1000);
pdf_vals = zeros(size(x_vals));
for i = 1:k
    pdf_vals = pdf_vals + weights(i) * normpdf(x_vals, means(i), sigmas(i));
end

plot(x_vals, pdf_vals, 'r-', 'LineWidth', 2);
xline(most_repetitive_value, '--k', 'Most Frequent');
title('Gaussian Mixture Model Fit');
xlabel('Data Value'); ylabel('Density');
legend('Histogram', 'GMM Fit', 'Most Repetitive Value');

if k == 1
    break;
end 


end
