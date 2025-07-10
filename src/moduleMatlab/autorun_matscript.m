close all;
clc;
clear;

data_dir_root = input("enter data_dir :","s"); 
%Example usage:  % without any quotes 
% enter data_dir :C:\Users\Gaetano\Desktop\create_with_codeRafi\MyProjects\Substructure_Different_DataTypes\results\kmeans_fgdata\

disp("Path entered: ");
disp(data_dir_root);

% data_dirParent = fullfile(data_dir_root,"data_dir");

% disp(data_dir_root); data_dir_root = fullfile(data_dir_root);

Prefix_intial = input("input prefix:","s");
% example usage: input prefix:coord_kmeans % without any quotes 

Prefix = Prefix_intial + "*" ;
disp(['Prefix pattern: ', Prefix])
disp(Prefix)
% only one arguments taken by the dir, it will return all files and folder in the given directory path those start with Prefix as entered by the user.
find_dirWithPrefix = dir(fullfile(data_dir_root,Prefix));   

%  # it returns the structre array with
size_structArray_WithPrefix= size(find_dirWithPrefix);
Totalfiles_withPrefix = size_structArray_WithPrefix(1,1);

disp(Totalfiles_withPrefix);

for i = 1:Totalfiles_withPrefix
    if isfolder(find_dirWithPrefix(i).folder)
        fprintf("\n loop no. --->  %d \t ",i);
        dir_name = find_dirWithPrefix(i).name ;
        dir_path = find_dirWithPrefix(i).folder ;
        fprintf("\n dir name: %s",dir_name);
        fprintf("\n dir path: %s ",dir_path);


        data_dir = fullfile(dir_path,dir_name);

% %         plotLabelsFromMatFolder(data_dir);
%             plotLabelsFromMatFolder_subplots(data_dir);


        %%% providing the input for the clusterlabelCoordPlot.m function.
               %%% data_dir = fullfile(dir_path,dir_name);
                fprintf("\n data_dir before feeding to .mat function: %s\n",data_dir);
                mode = 'point';  %  'mesh', 'point', or 'mesh+point'
                alpha_value = 5;
                min_cluster_size = 3;
                face_transparency = 0.6;
                use_subplots = false; % true
        
        
%                 clusterlabelCoordPlot_v2(data_dir, mode, alpha_value, min_cluster_size, face_transparency, use_subplots);

                clusterlabelCoordPlot_batch3(data_dir,mode, alpha_value, min_cluster_size, face_transparency, use_subplots)

%                 [~, name, ~] = fileparts(dir_name);
%                 saveas(gcf, fullfile(data_dir, [name '_labelPlot.png']));
%                 close(gcf);
%         %         
        %     % Print input arguments for debugging
        %     fprintf('\n[DEBUG] data_dir: %s\n', data_dir);
        %     fprintf('[DEBUG] mode: %s\n', mode);
        %     fprintf('[DEBUG] alpha_value: %.2f\n', alpha_value);
        %     fprintf('[DEBUG] min_cluster_size: %d\n', min_cluster_size);
        %     fprintf('[DEBUG] face_transparency: %.2f\n', face_transparency);
        %     fprintf('[DEBUG] use_subplots: %d\n', use_subplots);
    
    % --- continue with the rest of your function ---


        %       this part is valid when clusterlabelCoordPlot_v2 is defined to
        %         ask one .mat file each time then below for loop works. other just
        %         give .mat_filedirectory, it will process all .mat file
        %         automaticallyas it is defined to work with these type for:
        %         matfiles = dir(fullfile(dir_path,dir_name,"*.mat"));
        %         totalmatfiles = size(matfiles);
        %         totalMatFiles = totalmatfiles(1,1);

        %%         ii=1:totalMatFiles
        %             nameMatFile = matfiles(ii).name; %PathMatFile =
        %             matfiles(ii).folder ; PathMatFile =
        %             fullfile(dir_path,dir_name) ; fprintf("\n total mat files: %d
        %             \t and its name: %s\t  \n its full path: %s \n ",
        %             totalMatFiles,nameMatFile,PathMatFile);
        %
        %            clusterlabelCoordPlot_v2_single(mat_file_path, mode,
        %            alpha_value, min_cluster_size, face_transparency)
        %%        end

    end
    fprintf("\n finsihed with loop no. --->  %d \t \n ",i);
end

fprintf("\n  < -------------------- All processed finished ---> loop completed:-> %d\n ",i);

% files = dir(fullfile(data_dir_root,Prefix));

% disp(find_dirWithPrefix); disp(files);

% find_dirWithPrefix = dir(fullfile(data_dir_root,'coord_dbfinal*'));
