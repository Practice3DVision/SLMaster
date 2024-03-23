## 1. SLMaster
<img ref="https://github.com/Practice3DVision/SLMaster/releases/tag/v1.2.0" src="https://img.shields.io/badge/release-v1.2.0-blue" />
<img src="https://img.shields.io/badge/windows11-passing-rgb(0, 255, 0)" />
<img src="https://img.shields.io/badge/ubuntu22.04-Prepare for testing-rgb(255, 165, 0)" />

[SLMaster](https://github.com/Practice3DVision/SLMaster)是一个较为完整的结构光3D相机软件。您可以使用它对任意被测物体完成静态扫描或者实时动态扫描。

该软件将由博主的系列文章[从0到1搭建一套属于你自己的高精度实时结构光3D相机](https://mp.weixin.qq.com/s/E8K3892eNVJfgpMUHtf9Lw)手把手教学完成，可关注公众号进行学习。

请不要吝惜你的⭐，你的⭐和关注是博主源源不断的动力。有任何问题和**bug**反馈请提**Issue**。

**想先体验该软件？**

请在`Release`页面下载`exe`安装文件，但请注意`exe`安装方式不支持`GPU`加速。

## 2. 依赖
**SLMaster**依赖的库包含如下几点：
- [FluentUI](https://github.com/Practice3DVision/SLMaster/tree/master/FluentUI) <img src="https://img.shields.io/badge/项目内包含（部分代码进行了修改）-passing-rgb(0, 255, 0)" />
- [QuickQanava](https://github.com/cneben/QuickQanava/tree/2.4.1) <img src="https://img.shields.io/badge/项目内包含v2.4.1-passing-rgb(0, 255, 0)" />
- [MVSDK](https://www.irayple.com/cn/serviceSupport/downloadCenter/18?p=17) <img src="https://img.shields.io/badge/项目内包含v2.3.5-passing-rgb(0, 255, 0)" />
- [opencv_contribute](https://github.com/opencv/opencv_contrib.git) <img src="https://img.shields.io/badge/如果需要CUDA加速v4.8.0-passing-rgb(0, 255, 0)" />
- [OpenCV](https://github.com/opencv/opencv.git) <img src="https://img.shields.io/badge/v4.8.0-passing-rgb(0, 255, 0)" />
- [VTK](https://github.com/Kitware/VTK/tree/v9.2.0) <img src="https://img.shields.io/badge/v9.2.0-passing-rgb(0, 255, 0)" />
- [PCL](https://github.com/PointCloudLibrary/pcl/tree/pcl-1.12.1) <img src="https://img.shields.io/badge/v1.12.1-passing-rgb(0, 255, 0)" /> 
- [Qt5](https://doc.qt.io/qt-5/index.html) <img src="https://img.shields.io/badge/v5.15.14-passing-rgb(0, 255, 0)" />

> 如果电脑没有`NVIDIA GPU`，软件仍然能够使用CPU加速有效运行，此时可无需`opencv_contribute`依赖。

## 3. 编译
当你获取到本库代码之后，首先检查上述依赖，若不满足依赖条件，可通过点击上述依赖库跳转至对应的库，随后下载其代码并进行编译。以上面库皆没有编译安装的环境为例，它的编译顺序应该是这样的：

1. 下载[OpenCV](https://github.com/opencv/opencv.git)和[opencv_contribute](https://github.com/opencv/opencv_contrib.git)并进行编译（若`WITH_CUDA`未勾选请勾选上）
2. 下载[Qt5.15](https://doc.qt.io/qt-5/index.html)并选择`MSVC`编译套件安装
3. 下载[VTK](https://github.com/Kitware/VTK/tree/v9.2.0)并令`VTK_GROUP_ENABLE_Qt=YES`进行编译
4. 下载[PCL-1.12.1-AllInOne](https://github.com/PointCloudLibrary/pcl/releases)进行安装，安装完成后删除`PCL`安装文件夹下的除`3rdParty`外的其它任何文件，并将`3rdParty`文件夹中的`VTK`文件夹删除
5. 下载[PCL](https://github.com/PointCloudLibrary/pcl/tree/pcl-1.12.1)并选择好第三方库路径进行编译
6. 打开命令行窗口，键入`git clone --recursive https://github.com/Practice3DVision/SLMaster.git`克隆`SLMaster`
7. 打开`VSCode`编译运行`SLMaster`即可

> 注意！
> 每当编译好一个库都应当在系统环境变量中加入。例如，编译完成OpenCV后，设置好系统环境变量OpenCV_DIR路径。


你可以打开`SLMaster`中的`BUILD_TEST`选项，这将编译**google_test**中的测试用例，这些测试用例同样是一份非常不错的示例代码。

当你成功进行编译后，将得到可执行文件`SLMasterGui.exe`,它位于`build/gui/`目录之下。随后运行它尽情享受吧！
## 4. 使用

离线使用情况下，可通过进入`扫描模式->离线扫描模式->选择左相机文件夹->选择右相机文件夹->开始扫描->单次扫描`测试离线重建效果，软件提供一组离线数据集位于`安装目录/data/`下。

如您需要更改算法参数以测试自己的离线数据集，请通过更改`安装目录/gui/qml/res/config`下的相机配置文件，该文件记录了**3D**相机所有的状态，包括硬件组成、算法参数等。

如您需要接入硬件并执行在线功能，请修改`安装目录/gui/qml/res/config`下的相机配置文件，确保硬件组成参数与您所用的硬件设备一致。
## 5. 功能
**SLMaster**具备完整的软件功能，关键功能可以总结为以下几点：
- 单目/双目/三目结构光相机切换功能
- 3D相机在线/离线实时检测功能
- 3D相机连接/断开连接功能
- 光强、曝光时间调整功能
- 投影仪编码图案在线调试功能
  
  包括单次投影、连续投影、暂停投影、步进投影、停止投影等。
- 在线条纹编码功能
  
  包括1/8位深度图案，图案类型包括相移互补格雷码方案、多频外插方案（待完成）、多视立体几何方案，离焦方法包括传统二值离焦、二维误差扩散方法、最佳脉冲宽度调制方法。
- 在线条纹烧录功能
- 十字线校准功能
- 实时图像查看功能
- 离线相机标定功能
  
  包括单目/双目标定，受支持的标定板类型包括棋盘格、圆点、同心双圆环。此外，具备图像查看、显示误差分布图、显示极线校正图等功能。
- 在线投影仪标定功能
  
  可通过前述在线编码功能选择自己需要的相移互补格雷码周期数目，随后进入该功能执行在线投影仪标定，标定投影仪的内参和与相机的外参，支持单次内外参同时标定或先标内参再标外参。受支持的标定板类型包括棋盘格、圆点、同心双圆环。此外，具备显示误差分布图、水平绝对相位、垂直绝对相位、相机图像、投影仪映射图像查看等功能。
- 扫描模式切换功能（静态扫描/动态扫描/离线扫描）
- 实时三维点云渲染输出功能
- 实时纹理渲染功能
- 单次重建/实时重建功能
- 深度颜色归一化功能
- 点云裁剪功能
- 点查询功能
- 点云保存功能
- 帧率/点云数据量显示功能
- 点云后处理功能 
  
  该模块通过[QuickQanava](https://github.com/cneben/QuickQanava)实现的节点编辑器而实现。通过将各类**PCL**算子封装成节点，实现了以节点编辑器为基础，拖曳并连接节点而形成特定的点云处理算法。该种方式避免了在针对各类点云场景下，频繁的代码修改而耗时耗力的问题，可在线修改参数或是编辑算法流程以查看后处理效果。
- 英/汉语言切换
## 5. 示例
|功能|运行示例|
|-|-|
|十字线校准等功能|![十字线校准功能](doc/tenline.png)|
|条纹生成功能|![条纹生成功能](doc/stripe_create.png)|
|离线相机标定功能|![离线相机标定功能](doc/calibration.png)|
|标定误差分布图显示功能|![标定误差分布图显示功能](doc/error_distribute.png)|
|在线投影仪标定功能|![在线投影仪标定功能](doc/online_calinbration.png)
|在线扫描功能|![在线扫描功能](doc/online_scan.png)|
|实时扫描功能|![实时扫描功能](doc/04.gif)|
|后处理功能|![后处理功能](doc/post_process.png)|