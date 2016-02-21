caffe.set_mode_gpu()
caffe.reset_all()

avg = cat(3,122.67891434,116.66876762,104.00698793);

net = caffe.Net('/home/gipadmin/caffe_fc/models/fc8/fcn-8s-pascalcontext-deploy.prototxt',...
    '/home/gipadmin/caffe_fc/models/fc8/fcn-8s-pascalcontext.caffemodel','test');

%%
I=imread('people.jpg');
rsz = 100;
I = imresize(I,[rsz rsz]); I_=I;
subplot(121);
imshow(I)

I = single(I) - repmat(avg,[rsz rsz]);

I = I(:,:,[3 2 1]);
I = permute(I,[2 1 3]);

I1 = I;

n=100;
tic;
for i=1:n
    net.blobs('data').set_data(reshape(I1,[rsz rsz 3 1]));
    net.forward_prefilled();
    [m,ind1]=max(net.blobs('score').get_data(),[],3);
end
t2=toc;

fprintf('Total time: %f\n',(t2)/n);

ind = ind1;
subplot(122)

imagesc(ind')
axis equal;
axis tight;
axis off;
%r=rand(1,5)*10;
%imwrite(uint16(ind'),sprintf('outImages/TAG%.4i.tiff',i));
%imwrite(I_,sprintf('outImages/IM%.4i.png',i));

drawnow