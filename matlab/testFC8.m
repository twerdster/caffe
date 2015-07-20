caffe.set_mode_cpu()
caffe.reset_all()

avg = cat(3,122.67891434,116.66876762,104.00698793);

net = caffe.Net('/home/gipadmin/caffe_fc/models/fc8/fcn-8s-pascal-deploy.prototxt','/home/gipadmin/caffe_fc/models/fc8/fcn-8s-pascal.caffemodel','test');
vidFile = 'object.mp4';
vid = VideoReader(vidFile);

%I=imread('people.jpg');

for i=1:196
    I = read(vid,i);
    
    subplot(121)
    I = imresize(I,[500 1000]); I_=I;
    imshow(I_)
    
    I = single(I) - repmat(avg,[500 1000]);
    I = I(:,:,[3 2 1]);
    I = permute(I,[2 1 3]);
    
    I1 = I(1:500,:,:);
    net.blobs('data').set_data(reshape(I1,[500 500 3 1]));
    net.forward_prefilled();
    [m,ind1]=max(net.blobs('upscore').get_data(),[],3);
    
    I2 = I(501:end,:,:);
    net.blobs('data').set_data(reshape(I2,[500 500 3 1]));
    net.forward_prefilled();
    [m,ind2]=max(net.blobs('upscore').get_data(),[],3);
    
    ind = cat(1,ind1,ind2);
    subplot(122)
    
    imagesc(ind')
    axis equal;
    axis tight;
    axis off;
    imwrite(uint16(ind'),sprintf('outImages/TAG%.4i.tiff',i));
    imwrite(I_,sprintf('outImages/IM%.4i.png',i));
    
    drawnow
end