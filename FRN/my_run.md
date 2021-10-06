 我写的在colab中运行的代码

~~~
%cd /content
!pwd
%cd /content
# 下载代码
!git clone https://github.com/Jf-Chen/colabEdit.git
%cd /content/colabEdit/FRN/data
# 设置权限
!chmod 755 /content/colabEdit/FRN/data/download_mini_ImageNet.sh
# 下载数据，下载位置写在config.yml中
!/content/colabEdit/FRN/data/download_mini_ImageNet.sh

!pwd
%cd /content/colabEdit/FRN/data
!python /content/colabEdit/FRN/data/init_mini-ImageNet_my.py
# 到这一步数据预处理完成，得到了/content/colabEdit/FRN/data/mini-ImageNet
~~~

