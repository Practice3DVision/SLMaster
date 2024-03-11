#ifndef __IMV_DEFINES_H__
#define __IMV_DEFINES_H__

#ifdef WIN32
typedef __int64 int64_t;
typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

#ifndef IN
#define IN		///< \~chinese 输入型参数		\~english Input param 
#endif

#ifndef OUT
#define OUT		///< \~chinese 输出型参数		\~english Output param 
#endif

#ifndef IN_OUT
#define IN_OUT	///< \~chinese 输入/输出型参数  \~english Input/Output param 
#endif

#ifndef __cplusplus
typedef char    bool;
#define true    1
#define false   0
#endif

/// \~chinese
/// \brief 错误码 
/// \~english
/// \brief Error code
#define IMV_OK						0			///< \~chinese 成功，无错误							\~english Successed, no error
#define IMV_ERROR					-101		///< \~chinese 通用的错误							\~english Generic error
#define IMV_INVALID_HANDLE			-102		///< \~chinese 错误或无效的句柄						\~english Error or invalid handle
#define IMV_INVALID_PARAM			-103		///< \~chinese 错误的参数							\~english Incorrect parameter
#define IMV_INVALID_FRAME_HANDLE	-104		///< \~chinese 错误或无效的帧句柄					\~english Error or invalid frame handle
#define IMV_INVALID_FRAME			-105		///< \~chinese 无效的帧								\~english Invalid frame
#define IMV_INVALID_RESOURCE		-106		///< \~chinese 相机/事件/流等资源无效				\~english Camera/Event/Stream and so on resource invalid
#define IMV_INVALID_IP				-107		///< \~chinese 设备与主机的IP网段不匹配				\~english Device's and PC's subnet is mismatch
#define IMV_NO_MEMORY				-108		///< \~chinese 内存不足								\~english Malloc memery failed
#define IMV_INSUFFICIENT_MEMORY		-109		///< \~chinese 传入的内存空间不足					\~english Insufficient memory
#define IMV_ERROR_PROPERTY_TYPE		-110		///< \~chinese 属性类型错误							\~english Property type error
#define IMV_INVALID_ACCESS			-111		///< \~chinese 属性不可访问、或不能读/写、或读/写失败	\~english Property not accessible, or not be read/written, or read/written failed
#define IMV_INVALID_RANGE			-112		///< \~chinese 属性值超出范围、或者不是步长整数倍	\~english The property's value is out of range, or is not integer multiple of the step
#define IMV_NOT_SUPPORT				-113		///< \~chinese 设备不支持的功能						\~english Device not supported function

#define IMV_MAX_DEVICE_ENUM_NUM		100			///< \~chinese 支持设备最大个数		\~english The maximum number of supported devices
#define IMV_MAX_STRING_LENTH		256			///< \~chinese 字符串最大长度		\~english The maximum length of string
#define IMV_MAX_ERROR_LIST_NUM		128			///< \~chinese 失败属性列表最大长度 \~english The maximum size of failed properties list

typedef	void*	IMV_HANDLE;						///< \~chinese 设备句柄				\~english Device handle 
typedef	void*	IMV_FRAME_HANDLE;				///< \~chinese 帧句柄				\~english Frame handle 

/// \~chinese
///枚举：属性类型
/// \~english
///Enumeration: property type
typedef enum _IMV_EFeatureType
{
	featureInt = 0x10000000, 				///< \~chinese 整型数				\~english Integer
	featureFloat = 0x20000000,				///< \~chinese 浮点数				\~english Float
	featureEnum = 0x30000000,				///< \~chinese 枚举					\~english Enumeration
	featureBool = 0x40000000,				///< \~chinese 布尔					\~english Bool
	featureString = 0x50000000,				///< \~chinese 字符串				\~english String
	featureCommand = 0x60000000,			///< \~chinese 命令					\~english Command
	featureGroup = 0x70000000,				///< \~chinese 分组节点				\~english Group Node
	featureReg = 0x80000000,				///< \~chinese 寄存器节点			\~english Register Node

	featureUndefined = 0x90000000			///< \~chinese 未定义				\~english Undefined
}IMV_EFeatureType;

/// \~chinese
///枚举：接口类型
/// \~english
///Enumeration: interface type
typedef enum _IMV_EInterfaceType
{
	interfaceTypeGige = 0x00000001,			///< \~chinese 网卡接口类型  		\~english NIC type
	interfaceTypeUsb3 = 0x00000002,			///< \~chinese USB3.0接口类型		\~english USB3.0 interface type
	interfaceTypeCL = 0x00000004, 			///< \~chinese CAMERALINK接口类型	\~english Cameralink interface type
	interfaceTypePCIe = 0x00000008,			///< \~chinese PCIe接口类型         \~english PCIe interface type
	interfaceTypeAll = 0x00000000,			///< \~chinese 忽略接口类型			\~english All types interface type
	interfaceInvalidType = 0xFFFFFFFF		///< \~chinese 无效接口类型			\~english Invalid interface type
}IMV_EInterfaceType;

/// \~chinese
///枚举：设备类型
/// \~english
///Enumeration: device type
typedef enum _IMV_ECameraType
{
	typeGigeCamera = 0,						///< \~chinese GIGE相机				\~english GigE Vision Camera
	typeU3vCamera = 1,						///< \~chinese USB3.0相机			\~english USB3.0 Vision Camera
	typeCLCamera = 2,						///< \~chinese CAMERALINK 相机		\~english Cameralink camera
	typePCIeCamera = 3,						///< \~chinese PCIe相机				\~english PCIe Camera
	typeUndefinedCamera = 255				///< \~chinese 未知类型				\~english Undefined Camera
}IMV_ECameraType;

/// \~chinese
///枚举：创建句柄方式
/// \~english
///Enumeration: Create handle mode
typedef enum _IMV_ECreateHandleMode
{
	modeByIndex = 0,						///< \~chinese 通过已枚举设备的索引(从0开始，比如 0, 1, 2...)	\~english By index of enumerated devices (Start from 0, such as 0, 1, 2...)
	modeByCameraKey,						///< \~chinese 通过设备键"厂商:序列号"							\~english By device's key "vendor:serial number"
	modeByDeviceUserID,						///< \~chinese 通过设备自定义名									\~english By device userID
	modeByIPAddress,						///< \~chinese 通过设备IP地址									\~english By device IP address.
}IMV_ECreateHandleMode;

/// \~chinese
///枚举：访问权限
/// \~english
///Enumeration: access permission
typedef enum _IMV_ECameraAccessPermission
{
	accessPermissionOpen = 0,				///< \~chinese GigE相机没有被连接			\~english The GigE vision device isn't connected to any application. 
	accessPermissionExclusive,				///< \~chinese 独占访问权限					\~english Exclusive Access Permission   
	accessPermissionControl, 				///< \~chinese 非独占可读访问权限			\~english Non-Exclusive Readbale Access Permission  
	accessPermissionControlWithSwitchover,  ///< \~chinese 切换控制访问权限				\~english Control access with switchover enabled.	
	accessPermissionUnknown = 254,  		///< \~chinese 无法确定						\~english Value not known; indeterminate.   	
	accessPermissionUndefined				///< \~chinese 未定义访问权限				\~english Undefined Access Permission
}IMV_ECameraAccessPermission;

/// \~chinese
///枚举：抓图策略
/// \~english
///Enumeration: grab strartegy
typedef enum _IMV_EGrabStrategy
{
	grabStrartegySequential = 0,			///< \~chinese 按到达顺序处理图片	\~english The images are processed in the order of their arrival
	grabStrartegyLatestImage = 1,			///< \~chinese 获取最新的图片		\~english Get latest image
	grabStrartegyUpcomingImage = 2,			///< \~chinese 等待获取下一张图片(只针对GigE相机)	\~english Waiting for next image(GigE only)
	grabStrartegyUndefined   				///< \~chinese 未定义				\~english Undefined
}IMV_EGrabStrategy;

/// \~chinese
///枚举：流事件状态
/// \~english
/// Enumeration:stream event status
typedef enum _IMV_EEventStatus
{
	streamEventNormal = 1,						///< \~chinese 正常流事件		\~english Normal stream event
	streamEventLostFrame = 2,					///< \~chinese 丢帧事件		    \~english Lost frame event
	streamEventLostPacket = 3,					///< \~chinese 丢包事件		    \~english Lost packet event
	streamEventImageError = 4,					///< \~chinese 图像错误事件		\~english Error image event
	streamEventStreamChannelError = 5,			///< \~chinese 取流错误事件		\~english Stream channel error event
	streamEventTooManyConsecutiveResends = 6,	///< \~chinese 太多连续重传		\~english Too many consecutive resends event
	streamEventTooManyLostPacket = 7			///< \~chinese 太多丢包			\~english Too many lost packet event
}IMV_EEventStatus;

/// \~chinese
///枚举：图像转换Bayer格式所用的算法
/// \~english
/// Enumeration:alorithm used for Bayer demosaic
typedef enum _IMV_EBayerDemosaic
{
	demosaicNearestNeighbor,					///< \~chinese 最近邻			\~english Nearest neighbor
	demosaicBilinear,							///< \~chinese 双线性			\~english Bilinear
	demosaicEdgeSensing,						///< \~chinese 边缘检测			\~english Edge sensing
	demosaicNotSupport = 255,					///< \~chinese 不支持			\~english Not support
}IMV_EBayerDemosaic;

/// \~chinese
///枚举：事件类型
/// \~english
/// Enumeration:event type
typedef enum _IMV_EVType
{
	offLine,									///< \~chinese 设备离线通知		\~english device offline notification
	onLine										///< \~chinese 设备在线通知		\~english device online notification
}IMV_EVType;

/// \~chinese
///枚举：视频格式
/// \~english
/// Enumeration:Video format
typedef enum _IMV_EVideoType
{
	typeVideoFormatAVI = 0,						///< \~chinese AVI格式			\~english AVI format
	typeVideoFormatNotSupport = 255				///< \~chinese 不支持			\~english Not support
}IMV_EVideoType;

/// \~chinese
///枚举：图像翻转类型
/// \~english
/// Enumeration:Image flip type
typedef enum _IMV_EFlipType
{
	typeFlipVertical,							///< \~chinese 垂直(Y轴)翻转	\~english Vertical(Y-axis) flip
	typeFlipHorizontal							///< \~chinese 水平(X轴)翻转	\~english Horizontal(X-axis) flip
}IMV_EFlipType;

/// \~chinese
///枚举：顺时针旋转角度
/// \~english
/// Enumeration:Rotation angle clockwise
typedef enum _IMV_ERotationAngle
{
	rotationAngle90,							///< \~chinese 顺时针旋转90度	\~english Rotate 90 degree clockwise
	rotationAngle180,							///< \~chinese 顺时针旋转180度	\~english Rotate 180 degree clockwise
	rotationAngle270,							///< \~chinese 顺时针旋转270度	\~english Rotate 270 degree clockwise
}IMV_ERotationAngle;

#define IMV_GVSP_PIX_MONO                           0x01000000
#define IMV_GVSP_PIX_RGB                            0x02000000
#define IMV_GVSP_PIX_COLOR                          0x02000000
#define IMV_GVSP_PIX_CUSTOM                         0x80000000
#define IMV_GVSP_PIX_COLOR_MASK                     0xFF000000

// Indicate effective number of bits occupied by the pixel (including padding).
// This can be used to compute amount of memory required to store an image.
#define IMV_GVSP_PIX_OCCUPY1BIT                     0x00010000
#define IMV_GVSP_PIX_OCCUPY2BIT                     0x00020000
#define IMV_GVSP_PIX_OCCUPY4BIT                     0x00040000
#define IMV_GVSP_PIX_OCCUPY8BIT                     0x00080000
#define IMV_GVSP_PIX_OCCUPY12BIT                    0x000C0000
#define IMV_GVSP_PIX_OCCUPY16BIT                    0x00100000
#define IMV_GVSP_PIX_OCCUPY24BIT                    0x00180000
#define IMV_GVSP_PIX_OCCUPY32BIT                    0x00200000
#define IMV_GVSP_PIX_OCCUPY36BIT                    0x00240000
#define IMV_GVSP_PIX_OCCUPY48BIT                    0x00300000

/// \~chinese
///枚举：图像格式
/// \~english
/// Enumeration:image format
typedef enum _IMV_EPixelType
{
	// Undefined pixel type
	gvspPixelTypeUndefined = -1,

	// Mono Format
	gvspPixelMono1p = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY1BIT | 0x0037),
	gvspPixelMono2p = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY2BIT | 0x0038),
	gvspPixelMono4p = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY4BIT | 0x0039),
	gvspPixelMono8 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY8BIT | 0x0001),
	gvspPixelMono8S = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY8BIT | 0x0002),
	gvspPixelMono10 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0003),
	gvspPixelMono10Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x0004),
	gvspPixelMono12 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0005),
	gvspPixelMono12Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x0006),
	gvspPixelMono14 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0025),
	gvspPixelMono16 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0007),

	// Bayer Format
	gvspPixelBayGR8 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY8BIT | 0x0008),
	gvspPixelBayRG8 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY8BIT | 0x0009),
	gvspPixelBayGB8 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY8BIT | 0x000A),
	gvspPixelBayBG8 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY8BIT | 0x000B),
	gvspPixelBayGR10 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x000C),
	gvspPixelBayRG10 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x000D),
	gvspPixelBayGB10 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x000E),
	gvspPixelBayBG10 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x000F),
	gvspPixelBayGR12 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0010),
	gvspPixelBayRG12 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0011),
	gvspPixelBayGB12 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0012),
	gvspPixelBayBG12 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0013),
	gvspPixelBayGR10Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x0026),
	gvspPixelBayRG10Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x0027),
	gvspPixelBayGB10Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x0028),
	gvspPixelBayBG10Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x0029),
	gvspPixelBayGR12Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x002A),
	gvspPixelBayRG12Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x002B),
	gvspPixelBayGB12Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x002C),
	gvspPixelBayBG12Packed = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY12BIT | 0x002D),
	gvspPixelBayGR16 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x002E),
	gvspPixelBayRG16 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x002F),
	gvspPixelBayGB16 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0030),
	gvspPixelBayBG16 = (IMV_GVSP_PIX_MONO | IMV_GVSP_PIX_OCCUPY16BIT | 0x0031),

	// RGB Format
	gvspPixelRGB8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY24BIT | 0x0014),
	gvspPixelBGR8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY24BIT | 0x0015),
	gvspPixelRGBA8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY32BIT | 0x0016),
	gvspPixelBGRA8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY32BIT | 0x0017),
	gvspPixelRGB10 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x0018),
	gvspPixelBGR10 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x0019),
	gvspPixelRGB12 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x001A),
	gvspPixelBGR12 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x001B),
	gvspPixelRGB16 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x0033),
	gvspPixelRGB10V1Packed = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY32BIT | 0x001C),
	gvspPixelRGB10P32 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY32BIT | 0x001D),
	gvspPixelRGB12V1Packed = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY36BIT | 0X0034),
	gvspPixelRGB565P = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x0035),
	gvspPixelBGR565P = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0X0036),

	// YVR Format
	gvspPixelYUV411_8_UYYVYY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY12BIT | 0x001E),
	gvspPixelYUV422_8_UYVY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x001F),
	gvspPixelYUV422_8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x0032),
	gvspPixelYUV8_UYV = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY24BIT | 0x0020),
	gvspPixelYCbCr8CbYCr = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY24BIT | 0x003A),
	gvspPixelYCbCr422_8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x003B),
	gvspPixelYCbCr422_8_CbYCrY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x0043),
	gvspPixelYCbCr411_8_CbYYCrYY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY12BIT | 0x003C),
	gvspPixelYCbCr601_8_CbYCr = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY24BIT | 0x003D),
	gvspPixelYCbCr601_422_8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x003E),
	gvspPixelYCbCr601_422_8_CbYCrY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x0044),
	gvspPixelYCbCr601_411_8_CbYYCrYY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY12BIT | 0x003F),
	gvspPixelYCbCr709_8_CbYCr = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY24BIT | 0x0040),
	gvspPixelYCbCr709_422_8 = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x0041),
	gvspPixelYCbCr709_422_8_CbYCrY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY16BIT | 0x0045),
	gvspPixelYCbCr709_411_8_CbYYCrYY = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY12BIT | 0x0042),

	// RGB Planar
	gvspPixelRGB8Planar = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY24BIT | 0x0021),
	gvspPixelRGB10Planar = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x0022),
	gvspPixelRGB12Planar = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x0023),
	gvspPixelRGB16Planar = (IMV_GVSP_PIX_COLOR | IMV_GVSP_PIX_OCCUPY48BIT | 0x0024),

	//BayerRG10p和BayerRG12p格式，针对特定项目临时添加,请不要使用
	//BayerRG10p and BayerRG12p, currently used for specific project, please do not use them
	gvspPixelBayRG10p = 0x010A0058,
	gvspPixelBayRG12p = 0x010c0059,

	//mono1c格式，自定义格式
	//mono1c, customized image format, used for binary output
	gvspPixelMono1c = 0x012000FF,

	//mono1e格式，自定义格式，用来显示连通域
	//mono1e, customized image format, used for displaying connected domain
	gvspPixelMono1e = 0x01080FFF
}IMV_EPixelType;

/// \~chinese
/// \brief 字符串信息
/// \~english
/// \brief String information
typedef struct _IMV_String
{
	char str[IMV_MAX_STRING_LENTH];						///< \~chinese	字符串.长度不超过256  \~english Strings and the maximum length of strings is 255.
}IMV_String;

/// \~chinese
/// \brief GigE网卡信息
/// \~english
/// \brief GigE interface information
typedef struct _IMV_GigEInterfaceInfo
{
	char description[IMV_MAX_STRING_LENTH];				///< \~chinese  网卡描述信息		\~english Network card description
	char macAddress[IMV_MAX_STRING_LENTH];				///< \~chinese  网卡Mac地址			\~english Network card MAC Address
	char ipAddress[IMV_MAX_STRING_LENTH];				///< \~chinese  设备Ip地址			\~english Device ip Address
	char subnetMask[IMV_MAX_STRING_LENTH];				///< \~chinese  子网掩码			\~english SubnetMask
	char defaultGateWay[IMV_MAX_STRING_LENTH];			///< \~chinese  默认网关			\~english Default GateWay
	char chReserved[5][IMV_MAX_STRING_LENTH];			///< 保留							\~english Reserved field
}IMV_GigEInterfaceInfo;

/// \~chinese
/// \brief USB接口信息
/// \~english
/// \brief USB interface information
typedef struct _IMV_UsbInterfaceInfo
{
	char description[IMV_MAX_STRING_LENTH];				///< \~chinese  USB接口描述信息		\~english USB interface description
	char vendorID[IMV_MAX_STRING_LENTH];				///< \~chinese  USB接口Vendor ID	\~english USB interface Vendor ID
	char deviceID[IMV_MAX_STRING_LENTH];				///< \~chinese  USB接口设备ID		\~english USB interface Device ID
	char subsystemID[IMV_MAX_STRING_LENTH];				///< \~chinese  USB接口Subsystem ID	\~english USB interface Subsystem ID
	char revision[IMV_MAX_STRING_LENTH];				///< \~chinese  USB接口Revision		\~english USB interface Revision
	char speed[IMV_MAX_STRING_LENTH];					///< \~chinese  USB接口speed		\~english USB interface speed
	char chReserved[4][IMV_MAX_STRING_LENTH];			///< 保留							\~english Reserved field
}IMV_UsbInterfaceInfo;

/// \~chinese
/// \brief GigE设备信息
/// \~english
/// \brief GigE device information
typedef struct _IMV_GigEDeviceInfo
{
	/// \~chinese
	/// 设备支持的IP配置选项\n
	/// value:4 相机只支持LLA\n
	/// value:5 相机支持LLA和Persistent IP\n
	/// value:6 相机支持LLA和DHCP\n
	/// value:7 相机支持LLA、DHCP和Persistent IP\n
	/// value:0 获取失败
	/// \~english
	/// Supported IP configuration options of device\n
	/// value:4 Device supports LLA \n
	/// value:5 Device supports LLA and Persistent IP\n
	/// value:6 Device supports LLA and DHCP\n
	/// value:7 Device supports LLA, DHCP and Persistent IP\n
	/// value:0 Get fail
	unsigned int nIpConfigOptions;
	/// \~chinese
	/// 设备当前的IP配置选项\n
	/// value:4 LLA处于活动状态\n
	/// value:5 LLA和Persistent IP处于活动状态\n
	/// value:6 LLA和DHCP处于活动状态\n
	/// value:7 LLA、DHCP和Persistent IP处于活动状态\n
	/// value:0 获取失败
	/// \~english
	/// Current IP Configuration options of device\n
	/// value:4 LLA is active\n
	/// value:5 LLA and Persistent IP are active\n
	/// value:6 LLA and DHCP are active\n
	/// value:7 LLA, DHCP and Persistent IP are active\n
	/// value:0 Get fail
	unsigned int nIpConfigCurrent;
	unsigned int nReserved[3];						///< \~chinese 保留					\~english Reserved field

	char macAddress[IMV_MAX_STRING_LENTH];			///< \~chinese 设备Mac地址			\~english Device MAC Address
	char ipAddress[IMV_MAX_STRING_LENTH];			///< \~chinese 设备Ip地址			\~english Device ip Address
	char subnetMask[IMV_MAX_STRING_LENTH];			///< \~chinese 子网掩码				\~english SubnetMask
	char defaultGateWay[IMV_MAX_STRING_LENTH];		///< \~chinese 默认网关				\~english Default GateWay
	char protocolVersion[IMV_MAX_STRING_LENTH];		///< \~chinese 网络协议版本			\~english Net protocol version
	/// \~chinese
	/// Ip配置有效性\n
	/// Ip配置有效时字符串值"Valid"\n
	/// Ip配置无效时字符串值"Invalid On This Interface"
	/// \~english
	/// IP configuration valid\n
	/// String value is "Valid" when ip configuration valid\n
	/// String value is "Invalid On This Interface" when ip configuration invalid
	char ipConfiguration[IMV_MAX_STRING_LENTH];
	char chReserved[6][IMV_MAX_STRING_LENTH];		///< \~chinese 保留					\~english Reserved field

}IMV_GigEDeviceInfo;

/// \~chinese
/// \brief Usb设备信息
/// \~english
/// \brief Usb device information
typedef struct _IMV_UsbDeviceInfo
{
	bool bLowSpeedSupported;						///< \~chinese true支持，false不支持，其他值 非法。	\~english true support,false not supported,other invalid
	bool bFullSpeedSupported;						///< \~chinese true支持，false不支持，其他值 非法。	\~english true support,false not supported,other invalid
	bool bHighSpeedSupported;						///< \~chinese true支持，false不支持，其他值 非法。	\~english true support,false not supported,other invalid
	bool bSuperSpeedSupported;						///< \~chinese true支持，false不支持，其他值 非法。	\~english true support,false not supported,other invalid
	bool bDriverInstalled;							///< \~chinese true安装，false未安装，其他值 非法。	\~english true support,false not supported,other invalid
	bool boolReserved[3];							///< \~chinese 保留		
	unsigned int Reserved[4];						///< \~chinese 保留									\~english Reserved field

	char configurationValid[IMV_MAX_STRING_LENTH];	///< \~chinese 配置有效性							\~english Configuration Valid
	char genCPVersion[IMV_MAX_STRING_LENTH];		///< \~chinese GenCP 版本							\~english GenCP Version
	char u3vVersion[IMV_MAX_STRING_LENTH];			///< \~chinese U3V 版本号							\~english U3v Version
	char deviceGUID[IMV_MAX_STRING_LENTH];			///< \~chinese 设备引导号							\~english Device guid number                  
	char familyName[IMV_MAX_STRING_LENTH];			///< \~chinese 设备系列号							\~english Device serial number 
	char u3vSerialNumber[IMV_MAX_STRING_LENTH];		///< \~chinese 设备序列号							\~english Device SerialNumber
	char speed[IMV_MAX_STRING_LENTH];				///< \~chinese 设备传输速度							\~english Device transmission speed
	char maxPower[IMV_MAX_STRING_LENTH];			///< \~chinese 设备最大供电量						\~english Maximum power supply of device
	char chReserved[4][IMV_MAX_STRING_LENTH];		///< \~chinese 保留									\~english Reserved field

}IMV_UsbDeviceInfo;

/// \~chinese
/// \brief 设备通用信息
/// \~english
/// \brief Device general information
typedef struct _IMV_DeviceInfo
{
	IMV_ECameraType					nCameraType;								///< \~chinese 设备类别			\~english Camera type
	int								nCameraReserved[5];							///< \~chinese 保留				\~english Reserved field

	char							cameraKey[IMV_MAX_STRING_LENTH];			///< \~chinese 厂商:序列号		\~english Camera key
	char							cameraName[IMV_MAX_STRING_LENTH];			///< \~chinese 用户自定义名		\~english UserDefinedName
	char							serialNumber[IMV_MAX_STRING_LENTH];			///< \~chinese 设备序列号		\~english Device SerialNumber
	char							vendorName[IMV_MAX_STRING_LENTH];			///< \~chinese 厂商				\~english Camera Vendor
	char							modelName[IMV_MAX_STRING_LENTH];			///< \~chinese 设备型号			\~english Device model
	char							manufactureInfo[IMV_MAX_STRING_LENTH];		///< \~chinese 设备制造信息		\~english Device ManufactureInfo
	char							deviceVersion[IMV_MAX_STRING_LENTH];		///< \~chinese 设备版本			\~english Device Version
	char							cameraReserved[5][IMV_MAX_STRING_LENTH];	///< \~chinese 保留				\~english Reserved field
	union
	{
		IMV_GigEDeviceInfo			gigeDeviceInfo;								///< \~chinese  Gige设备信息	\~english Gige Device Information
		IMV_UsbDeviceInfo			usbDeviceInfo;								///< \~chinese  Usb设备信息		\~english Usb  Device Information
	}DeviceSpecificInfo;

	IMV_EInterfaceType				nInterfaceType;								///< \~chinese 接口类别			\~english Interface type
	int								nInterfaceReserved[5];						///< \~chinese 保留				\~english Reserved field
	char							interfaceName[IMV_MAX_STRING_LENTH];		///< \~chinese 接口名			\~english Interface Name
	char							interfaceReserved[5][IMV_MAX_STRING_LENTH];	///< \~chinese 保留				\~english Reserved field
	union
	{
		IMV_GigEInterfaceInfo		gigeInterfaceInfo;							///< \~chinese  GigE网卡信息	\~english Gige interface Information
		IMV_UsbInterfaceInfo		usbInterfaceInfo;							///< \~chinese  Usb接口信息		\~english Usb interface Information
	}InterfaceInfo;
}IMV_DeviceInfo;

/// \~chinese
/// \brief 加载失败的属性信息
/// \~english
/// \brief Load failed properties information
typedef struct _IMV_ErrorList
{
	unsigned int		nParamCnt;									///< \~chinese 加载失败的属性个数			\~english The count of load failed properties
	IMV_String			paramNameList[IMV_MAX_ERROR_LIST_NUM];		///< \~chinese 加载失败的属性集合，上限128	\~english Array of load failed properties, up to 128 
}IMV_ErrorList;

/// \~chinese
/// \brief 设备信息列表
/// \~english
/// \brief Device information list
typedef struct _IMV_DeviceList
{
	unsigned int		nDevNum;					///< \~chinese 设备数量									\~english Device Number
	IMV_DeviceInfo*		pDevInfo;					///< \~chinese 设备息列表(SDK内部缓存)，最多100设备		\~english Device information list(cached within the SDK), up to 100
}IMV_DeviceList;

/// \~chinese
/// \brief 连接事件信息
/// \~english
/// \brief connection event information
typedef struct _IMV_SConnectArg
{
	IMV_EVType			event;						///< \~chinese 事件类型							\~english event type
	unsigned int		nReserve[10];				///< \~chinese 预留字段							\~english Reserved field
}IMV_SConnectArg;

/// \~chinese
/// \brief 参数更新事件信息
/// \~english
/// \brief Updating parameters event information
typedef struct _IMV_SParamUpdateArg
{
	bool				isPoll;						///< \~chinese 是否是定时更新,true:表示是定时更新，false:表示非定时更新	\~english Update periodically or not. true:update periodically, true:not update periodically
	unsigned int		nReserve[10];				///< \~chinese 预留字段							\~english Reserved field
	unsigned int		nParamCnt;					///< \~chinese 更新的参数个数					\~english The number of parameters which need update
	IMV_String*			pParamNameList;				///< \~chinese 更新的参数名称集合(SDK内部缓存)	\~english Array of parameter's name which need to be updated(cached within the SDK)
}IMV_SParamUpdateArg;

/// \~chinese
/// \brief 流事件信息
/// \~english
/// \brief Stream event information
typedef struct _IMV_SStreamArg
{
	unsigned int		channel;					///< \~chinese 流通道号         \~english Channel no.
	uint64_t			blockId;					///< \~chinese 流数据BlockID    \~english Block ID of stream data
	uint64_t			timeStamp;					///< \~chinese 时间戳           \~english Event time stamp
	IMV_EEventStatus	eStreamEventStatus;			///< \~chinese 流事件状态码     \~english Stream event status code
	unsigned int		status;						///< \~chinese 事件状态错误码   \~english Status error code
	unsigned int		nReserve[9];				///< \~chinese 预留字段         \~english Reserved field
}IMV_SStreamArg;

/// \~chinese
/// 消息通道事件ID列表
/// \~english
/// message channel event id list
#define IMV_MSG_EVENT_ID_EXPOSURE_END			0x9001
#define IMV_MSG_EVENT_ID_FRAME_TRIGGER			0x9002
#define IMV_MSG_EVENT_ID_FRAME_START			0x9003
#define IMV_MSG_EVENT_ID_ACQ_START				0x9004
#define IMV_MSG_EVENT_ID_ACQ_TRIGGER			0x9005
#define IMV_MSG_EVENT_ID_DATA_READ_OUT			0x9006

/// \~chinese
/// \brief 消息通道事件信息
/// \~english
/// \brief Message channel event information
typedef struct _IMV_SMsgChannelArg
{
	unsigned short		eventId;					///< \~chinese 事件Id								\~english Event id
	unsigned short		channelId;					///< \~chinese 消息通道号							\~english Channel id
	uint64_t			blockId;					///< \~chinese 流数据BlockID						\~english Block ID of stream data
	uint64_t			timeStamp;					///< \~chinese 时间戳								\~english Event timestamp
	unsigned int		nReserve[8];				///< \~chinese 预留字段         					\~english Reserved field
	unsigned int		nParamCnt;					///< \~chinese 参数个数								\~english The number of parameters which need update
	IMV_String*			pParamNameList;				///< \~chinese 事件相关的属性名列集合(SDK内部缓存)	\~english Array of parameter's name which is related(cached within the SDK)
}IMV_SMsgChannelArg;

/// \~chinese
/// \brief Chunk数据信息
/// \~english
/// \brief Chunk data information
typedef struct _IMV_ChunkDataInfo
{
	unsigned int			chunkID;				///< \~chinese ChunkID									\~english ChunkID
	unsigned int			nParamCnt;				///< \~chinese 属性名个数								\~english The number of paramNames
	IMV_String*				pParamNameList;			///< \~chinese Chunk数据对应的属性名集合(SDK内部缓存)	\~english ParamNames Corresponding property name of chunk data(cached within the SDK)
}IMV_ChunkDataInfo;

/// \~chinese
/// \brief 帧图像信息
/// \~english
/// \brief The frame image information
typedef struct _IMV_FrameInfo
{
	uint64_t				blockId;				///< \~chinese 帧Id(仅对GigE/Usb/PCIe相机有效)					\~english The block ID(GigE/Usb/PCIe camera only)
	unsigned int			status;					///< \~chinese 数据帧状态(0是正常状态)							\~english The status of frame(0 is normal status)
	unsigned int			width;					///< \~chinese 图像宽度											\~english The width of image
	unsigned int			height;					///< \~chinese 图像高度											\~english The height of image
	unsigned int			size;					///< \~chinese 图像大小											\~english The size of image
	IMV_EPixelType			pixelFormat;			///< \~chinese 图像像素格式										\~english The pixel format of image
	uint64_t				timeStamp;				///< \~chinese 图像时间戳(仅对GigE/Usb/PCIe相机有效)			\~english The timestamp of image(GigE/Usb/PCIe camera only)
	unsigned int			chunkCount;				///< \~chinese 帧数据中包含的Chunk个数(仅对GigE/Usb相机有效)	\~english The number of chunk in frame data(GigE/Usb Camera Only)
	unsigned int			paddingX;				///< \~chinese 图像paddingX(仅对GigE/Usb/PCIe相机有效)			\~english The paddingX of image(GigE/Usb/PCIe camera only)
	unsigned int			paddingY;				///< \~chinese 图像paddingY(仅对GigE/Usb/PCIe相机有效)			\~english The paddingY of image(GigE/Usb/PCIe camera only)
	unsigned int			recvFrameTime;			///< \~chinese 图像在网络传输所用的时间(单位:微秒,非GigE相机该值为0)	\~english The time taken for the image to be transmitted over the network(unit:us, The value is 0 for non-GigE camera)
	unsigned int			nReserved[19];			///< \~chinese 预留字段											\~english Reserved field
}IMV_FrameInfo;

/// \~chinese
/// \brief 帧图像数据信息
/// \~english
/// \brief Frame image data information
typedef struct _IMV_Frame
{
	IMV_FRAME_HANDLE		frameHandle;			///< \~chinese 帧图像句柄(SDK内部帧管理用)						\~english Frame image handle(used for managing frame within the SDK)
	unsigned char*			pData;					///< \~chinese 帧图像数据的内存首地址							\~english The starting address of memory of image data
	IMV_FrameInfo			frameInfo;				///< \~chinese 帧信息											\~english Frame information
	unsigned int			nReserved[10];			///< \~chinese 预留字段											\~english Reserved field
}IMV_Frame;

/// \~chinese
/// \brief PCIE设备统计流信息
/// \~english
/// \brief PCIE device stream statistics information
typedef struct _IMV_PCIEStreamStatsInfo
{
	unsigned int			imageError;				///< \~chinese 图像错误的帧数			\~english  Number of images error frames
	unsigned int			lostPacketBlock;		///< \~chinese 丢包的帧数				\~english  Number of frames lost
	unsigned int			nReserved0[10];			///< \~chinese 预留						\~english  Reserved field

	unsigned int			imageReceived;			///< \~chinese 正常获取的帧数			\~english  Number of frames acquired
	double					fps;                    ///< \~chinese 帧率						\~english  Frame rate
	double					bandwidth;              ///< \~chinese 带宽(Mbps)				\~english  Bandwidth(Mbps)
	unsigned int			nReserved[8];           ///< \~chinese 预留						\~english  Reserved field
}IMV_PCIEStreamStatsInfo;

/// \~chinese
/// \brief U3V设备统计流信息
/// \~english
/// \brief U3V device stream statistics information
typedef struct _IMV_U3VStreamStatsInfo
{
	unsigned int			imageError;				///< \~chinese 图像错误的帧数			\~english  Number of images error frames
	unsigned int			lostPacketBlock;		///< \~chinese 丢包的帧数				\~english  Number of frames lost
	unsigned int			nReserved0[10];			///< \~chinese 预留						\~english  Reserved field

	unsigned int			imageReceived;			///< \~chinese 正常获取的帧数			\~english  Number of images error frames
	double					fps;                    ///< \~chinese 帧率						\~english  Frame rate
	double					bandwidth;              ///< \~chinese 带宽(Mbps)				\~english  Bandwidth(Mbps)
	unsigned int			nReserved[8];           ///< \~chinese 预留						\~english  Reserved field
}IMV_U3VStreamStatsInfo;

/// \~chinese
/// \brief Gige设备统计流信息
/// \~english
/// \brief Gige device stream statistics information
typedef struct _IMV_GigEStreamStatsInfo
{
	unsigned int			nReserved0[10];			///< \~chinese 预留						\~english  Reserved field

	unsigned int			imageError;				///< \~chinese 图像错误的帧数			\~english  Number of image error frames
	unsigned int			lostPacketBlock;		///< \~chinese 丢包的帧数				\~english  Number of frames lost
	unsigned int			nReserved1[4];			///< \~chinese 预留						\~english  Reserved field
	unsigned int			nReserved2[5];			///< \~chinese 预留						\~english  Reserved field

	unsigned int			imageReceived;			///< \~chinese 正常获取的帧数			\~english  Number of frames acquired
	double					fps;                    ///< \~chinese 帧率						\~english  Frame rate
	double					bandwidth;              ///< \~chinese 带宽(Mbps)				\~english  Bandwidth(Mbps)
	unsigned int			nReserved[4];			///< \~chinese 预留						\~english  Reserved field
}IMV_GigEStreamStatsInfo;

/// \~chinese
/// \brief 统计流信息
/// \~english
/// \brief Stream statistics information
typedef struct _IMV_StreamStatisticsInfo
{
	IMV_ECameraType			nCameraType;			///< \~chinese 设备类型				\~english  Device type

	union
	{
		IMV_PCIEStreamStatsInfo	pcieStatisticsInfo;	///< \~chinese PCIE设备统计信息		\~english  PCIE device statistics information
		IMV_U3VStreamStatsInfo	u3vStatisticsInfo;	///< \~chinese U3V设备统计信息		\~english  U3V device statistics information
		IMV_GigEStreamStatsInfo	gigeStatisticsInfo;	///< \~chinese Gige设备统计信息		\~english  GIGE device statistics information
	};
}IMV_StreamStatisticsInfo;

/// \~chinese
/// \brief 枚举属性的枚举值信息
/// \~english
/// \brief Enumeration property 's enumeration value information
typedef struct _IMV_EnumEntryInfo
{
	uint64_t				value;							///< \~chinese 枚举值				\~english  Enumeration value 
	char					name[IMV_MAX_STRING_LENTH];		///< \~chinese symbol名				\~english  Symbol name
}IMV_EnumEntryInfo;

/// \~chinese
/// \brief 枚举属性的可设枚举值列表信息
/// \~english
/// \brief Enumeration property 's settable enumeration value list information
typedef struct _IMV_EnumEntryList
{
	unsigned int			nEnumEntryBufferSize;		///< \~chinese 存放枚举值内存大小					\~english The size of saving enumeration value 
	IMV_EnumEntryInfo*		pEnumEntryInfo;				///< \~chinese 存放可设枚举值列表(调用者分配缓存)	\~english Save the list of settable enumeration value(allocated cache by the caller)
}IMV_EnumEntryList;

/// \~chinese
/// \brief 像素转换结构体
/// \~english
/// \brief Pixel convert structure
typedef struct _IMV_PixelConvertParam
{
	unsigned int			nWidth;							///< [IN]	\~chinese 图像宽						\~english Width
	unsigned int			nHeight;						///< [IN]	\~chinese 图像高						\~english Height
	IMV_EPixelType			ePixelFormat;					///< [IN]	\~chinese 像素格式						\~english Pixel format
	unsigned char*			pSrcData;						///< [IN]	\~chinese 输入图像数据					\~english Input image data
	unsigned int			nSrcDataLen;					///< [IN]	\~chinese 输入图像长度					\~english Input image length
	unsigned int			nPaddingX;						///< [IN]	\~chinese 图像宽填充					\~english Padding X
	unsigned int			nPaddingY;						///< [IN]	\~chinese 图像高填充					\~english Padding Y
	IMV_EBayerDemosaic		eBayerDemosaic;					///< [IN]	\~chinese 转换Bayer格式算法				\~english Alorithm used for Bayer demosaic
	IMV_EPixelType			eDstPixelFormat;				///< [IN]	\~chinese 目标像素格式					\~english Destination pixel format
	unsigned char*			pDstBuf;						///< [OUT]	\~chinese 输出数据缓存(调用者分配缓存)	\~english Output data buffer(allocated cache by the caller)
	unsigned int			nDstBufSize;					///< [IN]   \~chinese 提供的输出缓冲区大小  		\~english Provided output buffer size
	unsigned int			nDstDataLen;					///< [OUT]	\~chinese 输出数据长度          		\~english Output data length
	unsigned int			nReserved[8];					///<		\~chinese 预留							\~english Reserved field
}IMV_PixelConvertParam;

/// \~chinese
/// \brief 录像结构体
/// \~english
/// \brief Record structure
typedef struct _IMV_RecordParam
{
	unsigned int			nWidth;							///< [IN]	\~chinese 图像宽						\~english Width
	unsigned int			nHeight;						///< [IN]	\~chinese 图像高						\~english Height
	float                   fFameRate;						///< [IN]	\~chinese 帧率(大于0)					\~english Frame rate(greater than 0)
	unsigned int            nQuality;						///< [IN]	\~chinese 视频质量(1-100)				\~english Video quality(1-100)
	IMV_EVideoType			recordFormat;					///< [IN]	\~chinese 视频格式						\~english Video format
	const char*				pRecordFilePath;				///< [IN]	\~chinese 保存视频路径          		\~english Save video path
	unsigned int            nReserved[5];					///<		\~chinese 预留							\~english Reserved
}IMV_RecordParam;

/// \~chinese
/// \brief 录像用帧信息结构体
/// \~english
/// \brief Frame information for recording structure
typedef struct _IMV_RecordFrameInfoParam
{
	unsigned char*			pData;							///< [IN]	\~chinese 图像数据						\~english Image data
	unsigned int			nDataLen;						///< [IN]	\~chinese 图像数据长度					\~english Image data length
	unsigned int			nPaddingX;						///< [IN]	\~chinese 图像宽填充					\~english Padding X
	unsigned int			nPaddingY;						///< [IN]	\~chinese 图像高填充					\~english Padding Y
	IMV_EPixelType			ePixelFormat;					///< [IN]	\~chinese 像素格式						\~english Pixel format
	unsigned int            nReserved[5];					///<		\~chinese 预留							\~english Reserved
}IMV_RecordFrameInfoParam;

/// \~chinese
/// \brief 图像翻转结构体
/// \~english
/// \brief Flip image structure
typedef struct _IMV_FlipImageParam
{
	unsigned int			nWidth;							///< [IN]	\~chinese 图像宽						\~english Width
	unsigned int			nHeight;						///< [IN]	\~chinese 图像高						\~english Height
	IMV_EPixelType			ePixelFormat;					///< [IN]	\~chinese 像素格式						\~english Pixel format
	IMV_EFlipType			eFlipType;						///< [IN]	\~chinese 翻转类型						\~english Flip type
	unsigned char*			pSrcData;						///< [IN]	\~chinese 输入图像数据					\~english Input image data
	unsigned int			nSrcDataLen;					///< [IN]	\~chinese 输入图像长度					\~english Input image length
	unsigned char*			pDstBuf;						///< [OUT]	\~chinese 输出数据缓存(调用者分配缓存)	\~english Output data buffer(allocated cache by the caller)
	unsigned int			nDstBufSize;					///< [IN]   \~chinese 提供的输出缓冲区大小  		\~english Provided output buffer size
	unsigned int			nDstDataLen;					///< [OUT]	\~chinese 输出数据长度          		\~english Output data length
	unsigned int            nReserved[8];					///<		\~chinese 预留							\~english Reserved
}IMV_FlipImageParam;

/// \~chinese
/// \brief 图像旋转结构体
/// \~english
/// \brief Rotate image structure
typedef struct _IMV_RotateImageParam
{
	unsigned int			nWidth;							///< [IN][OUT]	\~chinese 图像宽						\~english Width
	unsigned int			nHeight;						///< [IN][OUT]	\~chinese 图像高						\~english Height
	IMV_EPixelType			ePixelFormat;					///< [IN]		\~chinese 像素格式						\~english Pixel format
	IMV_ERotationAngle		eRotationAngle;					///< [IN]		\~chinese 旋转角度						\~english Rotation angle
	unsigned char*			pSrcData;						///< [IN]		\~chinese 输入图像数据					\~english Input image data
	unsigned int			nSrcDataLen;					///< [IN]		\~chinese 输入图像长度					\~english Input image length
	unsigned char*			pDstBuf;						///< [OUT]		\~chinese 输出数据缓存(调用者分配缓存)	\~english Output data buffer(allocated cache by the caller)
	unsigned int			nDstBufSize;					///< [IN]		\~chinese 提供的输出缓冲区大小  		\~english Provided output buffer size
	unsigned int			nDstDataLen;					///< [OUT]		\~chinese 输出数据长度          		\~english Output data length
	unsigned int			nReserved[8];					///<			\~chinese 预留							\~english Reserved
}IMV_RotateImageParam;

/// \~chinese
/// \brief 设备连接状态事件回调函数声明
/// \param pParamUpdateArg [in] 回调时主动推送的设备连接状态事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of device connection status event 
/// \param pStreamArg [in] The device connection status event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void(*IMV_ConnectCallBack)(const IMV_SConnectArg* pConnectArg, void* pUser);

/// \~chinese
/// \brief 参数更新事件回调函数声明
/// \param pParamUpdateArg [in] 回调时主动推送的参数更新事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of parameter update event
/// \param pStreamArg [in] The parameter update event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void(*IMV_ParamUpdateCallBack)(const IMV_SParamUpdateArg* pParamUpdateArg, void* pUser);

/// \~chinese
/// \brief 流事件回调函数声明
/// \param pStreamArg [in] 回调时主动推送的流事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of stream event
/// \param pStreamArg [in] The stream event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void(*IMV_StreamCallBack)(const IMV_SStreamArg* pStreamArg, void* pUser);

/// \~chinese
/// \brief 消息通道事件回调函数声明
/// \param pMsgChannelArg [in] 回调时主动推送的消息通道事件信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of message channel event
/// \param pMsgChannelArg [in] The message channel event which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void(*IMV_MsgChannelCallBack)(const IMV_SMsgChannelArg* pMsgChannelArg, void* pUser);

/// \~chinese
/// \brief 帧数据信息回调函数声明
/// \param pFrame [in] 回调时主动推送的帧信息
/// \param pUser [in] 用户自定义数据
/// \~english
/// \brief Call back function declaration of frame data information
/// \param pFrame [in] The frame information which will be active pushed out during the callback
/// \param pUser [in] User defined data
typedef void(*IMV_FrameCallBack)(IMV_Frame* pFrame, void* pUser);

#endif // __IMV_DEFINES_H__