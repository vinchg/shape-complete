clear all
close all
clc
for num = 0:10
    file_name = 'prediction_';
    appended = num2str(num);
    file_name_no = strcat(file_name,appended);
    file_name = strcat(file_name_no,'.npy');
    f = fopen(file_name);
    %dim = fread(f, [1, 3], 'uint64')
    o = fread(f, [1, 16], 'uint64');
    format long
    m_1 = fread(f, 32768, 'float');
    dim = [32, 32, 32];
    %for ind = 1:length(m_1)
    %    if  -0.001 < m_1(ind) && m_1(ind)< 0.001
    %        m_1(ind) = 3;
    %    end
    %end

    m = reshape(m_1, [dim(1), dim(2), dim(3)]);
    fclose(f);
    [X, Y, Z] = meshgrid(1:dim(1), 1:dim(2), 1:dim(3));
    p = patch(isosurface(X, Y, Z, m, .7));
    %p = patch(isosurface(m, 10));
    %now to convert tsdf to volume and mesh
    %isonormals(X,Y,Z,m,p)
    %p.FaceColor = 'red';

    l = light('Position',[0.6 0.5 0.5],'Style','infinite');


    p.FaceAlpha = 1;           % remove the transparency
    p.FaceColor = 'red';    % set the face colors to be interpolated
    p.LineStyle = 'none';      % remove the lines
    daspect([1 1 1])
    material 'shiny'
    view(3); 
    %axis tight
    camlight 
    lighting gouraud
    %signed distance field
    saveas(p,strcat(file_name_no,'.fig'));
    saveas(p,strcat(file_name_no,'.png'));
    clear all
    close all
end