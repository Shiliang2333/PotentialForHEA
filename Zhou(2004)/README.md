# Zhou2004
## 类型
EAM
## 发布年份
2004
## 描述
高熵合金MD模拟应用最广泛的势函数之一。本库文件现已支持Cu,Ag,Au,Ni,Pd,Pt,Al,Pb,Fe,Mo,Ta,W,Mg,Co,Ti,Zr,Cr,V,Nb,Hf共20种元素，其中前17个已被收录在[lammps官方仓库](https://github.com/lammps/lammps/tree/develop/tools/eam_database)中，后4个的参数分别取自文献2，3，4和5。
## 用法
使用命令行进入势函数文件所在路径，键入命令：`python create_eam.py -n * *`即可，其中*代表元素符号。例如，要生成CoCrNi的势函数，则键入：`python create_eam.py -n Co Cr Ni`。  
详情参考这个链接的[readme文件](https://github.com/lammps/lammps/tree/develop/tools/eam_database)。  
若需要平均原子势，可以参考程序[mdapy](https://mdapy.readthedocs.io/en/latest/)。  
## 原文链接
1.Cu,Ag,Au,Ni,Pd,Pt,Al,Pb,Fe,Mo,Ta,W,Mg,Co,Ti,Zr[参数](https://doi.org/10.1103/PhysRevB.69.144113).  
2.Cr[参数](https://doi.org/10.1103/PhysRevB.77.214108).  
3.V[参数](https://doi.org/10.1016/j.actamat.2021.117233).  
4.Nb[参数](https://doi.org/10.1088/0953-8984/25/20/209501).  
5.Hf[参数](https://doi.org/10.1021/acs.chemmater.8b03969).  