# 我自己写的，只下载mini_ImageNet
#!/bin/bash

. ../utils/parse_yaml.sh
. ../utils/gdownload.sh
. ../utils/conditional.sh

eval $(parse_yaml ../config.yml)
echo 'this is the data_path you are trying to download data into:'
echo $data_path
mycwd=$(pwd)

cd $data_path

#---------这一段因为没有访问权限，需要从别的地方下载--------------#
# this section is for downloading images for mini-ImageNet
# credit to https://github.com/mileyan/simple_shot, https://github.com/twitter/meta-learning-lstm
# md5sum for the downloaded images.zip should be 987d2dfede486f633ec052ff463b62c6
#echo "downloading images for mini-ImageNet..."
#gdownload 0B3Irx3uQNoBMQ1FlNXJsZUdYWEE images.zip
#conditional_unzip images.zip 987d2dfede486f633ec052ff463b62c6

#---------我写的--------------#
# 数据来自https://mtl.yyliu.net/download/Lmzjm9tX.html 
# test.rar https://drive.google.com/file/d/1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v/view?usp=sharing
# train.rar https://drive.google.com/file/d/107FTosYIeBn5QbynR46YG91nHcJ70whs/view?usp=sharing
# val.rar https://drive.google.com/file/d/1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl/view?usp=sharing
echo "downloading images for mini-ImageNet..." 
gdownload 1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v test.tar
if [ ! -d "images" ]
then
	mkdir images
	echo "create dir images success" 
fi
cd images
tar -zxvf test.tar --strip-compontents 1