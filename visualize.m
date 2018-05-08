clear all
close all
clc
f = fopen('prediction_0.npy');
%dim = fread(f, [1, 3], 'uint64')
o = fread(f, [1, 16], 'uint64');
format long
m_1 = fread(f, 32768, 'float');
dim = [32, 32, 32];
for ind = 1:length(m_1)
    if  -0.01 < m_1(ind) && m_1(ind)< 0.01
        m_1(ind) = 3;
    end
end

m = reshape(m_1, [dim(1), dim(2), dim(3)]);
fclose(f);
[X, Y, Z] = meshgrid(1:dim(1), 1:dim(2), 1:dim(3));
p = patch(isosurface(X, Y, Z, m, .7));
%p = patch(isosurface(m, 10));
%now to convert tsdf to volume and mesh
%isonormals(X,Y,Z,m,p)
p.FaceColor = 'red';
p.EdgeColor = 'red';
daspect([1 1 1])
view(3); 
%axis tight
camlight 
lighting gouraud
%signed distance field