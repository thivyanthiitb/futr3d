wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval01_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval02_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval03_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval04_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval05_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval06_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval07_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval08_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval09_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval10_blobs.tgz
wget https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-test_blobs.tgz

pv v1.0-trainval01_blobs.tgz | tar -xz -C ./nuscenes
