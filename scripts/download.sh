wget https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/data.tar.gz
wget https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x00
wget https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x01
wget https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x02
wget https://github.com/XingangPan/GAN2Shape/releases/download/v1.0/x03

cat x0* > checkpoints.tar.gz
tar zxf data.tar.gz
tar zxf checkpoints.tar.gz
rm data.tar.gz
rm checkpoints.tar.gz
rm x0*