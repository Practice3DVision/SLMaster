/// \mainpage
/// \~chinese
/// \htmlinclude mainpage_chs.html
/// \~english
/// \htmlinclude mainpage_eng.html

#ifndef __IMV_API_H__
#define __IMV_API_H__

#include "IMVDefines.h"

/// \~chinese
/// \brief 动态库导入导出定义
/// \~english
/// \brief Dynamic library import and export definition
#if (defined (_WIN32) || defined(WIN64))
	#ifdef IMV_API_DLL_BUILD
		#define IMV_API _declspec(dllexport)
	#else
		#define IMV_API _declspec(dllimport)
	#endif

	#define IMV_CALL __stdcall
#else
	#define IMV_API
	#define IMV_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif 

/// \~chinese
/// \brief 获取版本信息
/// \return 成功时返回版本信息，失败时返回NULL
/// \~english
/// \brief get version information
/// \return Success, return version info. Failure, return NULL 
IMV_API const char* IMV_CALL IMV_GetVersion(void);

/// \~chinese
/// \brief 枚举设备
/// \param pDeviceList [OUT] 设备列表
/// \param interfaceType [IN] 待枚举的接口类型, 类型可任意组合,如 interfaceTypeGige | interfaceTypeUsb3
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 1、当interfaceType = interfaceTypeAll  时，枚举所有接口下的在线设备\n
/// 2、当interfaceType = interfaceTypeGige 时，枚举所有GigE网口下的在线设备\n
/// 3、当interfaceType = interfaceTypeUsb3 时，枚举所有USB接口下的在线设备\n
/// 4、当interfaceType = interfaceTypeCL   时，枚举所有CameraLink接口下的在线设备\n
/// 5、该接口下的interfaceType支持任意接口类型的组合,如，若枚举所有GigE网口和USB3接口下的在线设备时,
/// 可将interfaceType设置为 interfaceType = interfaceTypeGige | interfaceTypeUsb3,其它接口类型组合以此类推
/// \~english
/// \brief Enumerate Device
/// \param pDeviceList [OUT] Device list
/// \param interfaceType [IN] The interface type you want to find, support any interface type combination, sucn as interfaceTypeGige | interfaceTypeUsb3
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// 1、when interfaceType = interfaceTypeAll, enumerate devices in all interface types\n
/// 2、when interfaceType = interfaceTypeGige, enumerate devices in GigE interface \n
/// 3、when interfaceType = interfaceTypeUsb3, enumerate devices in USB interface\n
/// 4、when interfaceType = interfaceTypeCL, enumerate devices in CameraLink interface\n
/// 5、interfaceType supports any interface type combination. For example, if you want to find all GigE and USB3 devices,
/// you can set interfaceType as interfaceType = interfaceTypeGige | interfaceTypeUsb3.
IMV_API int IMV_CALL IMV_EnumDevices(OUT IMV_DeviceList *pDeviceList, IN unsigned int interfaceType);

/// \~chinese
/// \brief 以单播形式枚举设备, 仅限Gige设备使用
/// \param pDeviceList [OUT] 设备列表
/// \param pIpAddress [IN] 设备的IP地址
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Enumerate device by unicast mode. Only for Gige device.
/// \param pDeviceList [OUT] Device list
/// \param pIpAddress [IN] IP address of the device
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_EnumDevicesByUnicast(OUT IMV_DeviceList *pDeviceList, IN const char* pIpAddress);

/// \~chinese
/// \brief 通过指定标示符创建设备句柄，如指定索引、设备键、设备自定义名、IP地址
/// \param handle [OUT] 设备句柄
/// \param mode [IN] 创建设备方式
/// \param pIdentifier [IN] 指定标示符(设备键、设备自定义名、IP地址为char类型指针强转void类型指针，索引为unsigned int类型指针强转void类型指针)
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Create device handle by specifying identifiers, such as specifying index, device key, device userID, and IP address
/// \param handle [OUT] Device handle
/// \param mode [IN] Create handle mode
/// \param pIdentifier [IN] Specifying identifiers（device key, device userID, and IP address is char* forced to void*, index is unsigned int* forced to void*）
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_CreateHandle(OUT IMV_HANDLE* handle, IN IMV_ECreateHandleMode mode, IN void* pIdentifier);

/// \~chinese
/// \brief 销毁设备句柄
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Destroy device handle
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_DestroyHandle(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 获取设备信息
/// \param handle [IN] 设备句柄
/// \param pDevInfo [OUT] 设备信息
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get device information
/// \param handle [IN] Device handle
/// \param pDevInfo [OUT] Device information
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetDeviceInfo(IN IMV_HANDLE handle, OUT IMV_DeviceInfo *pDevInfo);

/// \~chinese
/// \brief  打开设备
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief  Open Device
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_Open(IN IMV_HANDLE handle);

/// \~chinese
/// \brief  打开设备
/// \param handle [IN] 设备句柄
/// \param accessPermission [IN] 控制通道权限(IMV_Open默认使用accessPermissionControl权限)
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief  Open Device
/// \param handle [IN] Device handle
/// \param accessPermission [IN] Control access permission(Default used accessPermissionControl in IMV_Open)
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_OpenEx(IN IMV_HANDLE handle, IN IMV_ECameraAccessPermission accessPermission);

/// \~chinese
/// \brief 判断设备是否已打开
/// \param handle [IN] 设备句柄
/// \return 打开状态，返回true；关闭状态或者掉线状态，返回false
/// \~english
/// \brief Check whether device is opened or not
/// \param handle [IN] Device handle
/// \return Opened, return true. Closed or Offline, return false 
IMV_API bool IMV_CALL IMV_IsOpen(IN IMV_HANDLE handle);

/// \~chinese
/// \brief  关闭设备
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Close Device
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_Close(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 修改设备IP, 仅限Gige设备使用
/// \param handle [IN] 设备句柄
/// \param pIpAddress [IN] IP地址
/// \param pSubnetMask [IN] 子网掩码
/// \param pGateway [IN] 默认网关
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 1、调用该函数时如果pSubnetMask和pGateway都设置了有效值，则以此有效值为准;\n
/// 2、调用该函数时如果pSubnetMask和pGateway都设置了NULL，则内部实现时用它所连接网卡的子网掩码和网关代替\n
/// 3、调用该函数时如果pSubnetMask和pGateway两者中其中一个为NULL，另一个非NULL，则返回错误
/// \~english
/// \brief Modify device IP. Only for Gige device.
/// \param handle [IN] Device handle
/// \param pIpAddress [IN] IP address 
/// \param pSubnetMask [IN] SubnetMask
/// \param pGateway [IN] Gateway
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// 1、When callback this function, if the values of pSubnetMask and pGateway are both valid then we consider the value is correct\n
/// 2、When callback this function, if the values of pSubnetMask and pGateway are both NULL, 
/// then these values will be replaced by the subnetmask and gateway of NIC which this device connect to.\n
/// 3、When callback this function, if there is one value of pSubnetMask or pGateway is NULL and the other one is not NULL, then return error
IMV_API int IMV_CALL IMV_GIGE_ForceIpAddress(IN IMV_HANDLE handle, IN const char* pIpAddress, IN const char* pSubnetMask, IN const char* pGateway);

/// \~chinese
/// \brief 获取设备的当前访问权限, 仅限Gige设备使用
/// \param handle [IN] 设备句柄
/// \param pAccessPermission [OUT] 设备的当前访问权限
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get current access permission of device. Only for Gige device.
/// \param handle [IN] Device handle
/// \param pAccessPermission [OUT] Current access permission of device
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GIGE_GetAccessPermission(IN IMV_HANDLE handle, OUT IMV_ECameraAccessPermission* pAccessPermission);

/// \~chinese
/// \brief 设置设备对sdk命令的响应超时时间,仅限Gige设备使用
/// \param handle [IN] 设备句柄
/// \param timeout [IN] 超时时间，单位ms
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Set the response timeout interval of device sends command to the API. Only for Gige device.
/// \param handle [IN] Device handle
/// \param timeout [IN] time out, unit：ms
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GIGE_SetAnswerTimeout(IN IMV_HANDLE handle, IN unsigned int timeout);

/// \~chinese
/// \brief 下载设备描述XML文件，并保存到指定路径，如：D:\\xml.zip
/// \param handle [IN] 设备句柄
/// \param pFullFileName [IN] 文件要保存的路径
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Download device description XML file, and save the files to specified path.  e.g. D:\\xml.zip
/// \param handle [IN] Device handle
/// \param pFullFileName [IN] The full paths where the downloaded XMl files would be saved to
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_DownLoadGenICamXML(IN IMV_HANDLE handle, IN const char* pFullFileName);

/// \~chinese
/// \brief 保存设备配置到指定的位置。同名文件已存在时，覆盖。
/// \param handle [IN] 设备句柄
/// \param pFullFileName [IN] 导出的设备配置文件全名(含路径)，如：D:\\config.xml 或 D:\\config.mvcfg
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english 
/// \brief Save the configuration of the device. Overwrite the file if exists.
/// \param handle [IN] Device handle
/// \param pFullFileName [IN] The full path name of the property file(xml).  e.g. D:\\config.xml or D:\\config.mvcfg
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_SaveDeviceCfg(IN IMV_HANDLE handle, IN const char* pFullFileName);

/// \~chinese
/// \brief 从文件加载设备xml配置
/// \param handle [IN] 设备句柄
/// \param pFullFileName [IN] 设备配置(xml)文件全名(含路径)，如：D:\\config.xml 或 D:\\config.mvcfg
/// \param pErrorList [OUT] 加载失败的属性名列表。存放加载失败的属性上限为IMV_MAX_ERROR_LIST_NUM。
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english 
/// \brief load the configuration of the device
/// \param handle [IN] Device handle
/// \param pFullFileName [IN] The full path name of the property file(xml). e.g. D:\\config.xml or D:\\config.mvcfg
/// \param pErrorList [OUT] The list of load failed properties. The failed to load properties list up to IMV_MAX_ERROR_LIST_NUM.
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_LoadDeviceCfg(IN IMV_HANDLE handle, IN const char* pFullFileName, OUT IMV_ErrorList* pErrorList);

/// \~chinese
/// \brief 写用户自定义数据。相机内部保留32768字节用于用户存储自定义数据(此功能针对本品牌相机，其它品牌相机无此功能)
/// \param handle [IN] 设备句柄
/// \param pBuffer [IN] 数据缓冲的指针
/// \param pLength [IN] 期望写入的字节数 [OUT] 实际写入的字节数
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english 
/// \brief Write user-defined data; Inside the camera, there are 32768 bytes reserved for user to store private data (Only for our own camera has this function)
/// \param handle [IN] Device handle
/// \param pBuffer [IN] Pointer of the data buffer
/// \param pLength [IN] Byte count written expected   [OUT] Byte count written in fact
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_WriteUserPrivateData(IN IMV_HANDLE handle, IN void* pBuffer, IN_OUT unsigned int* pLength);

/// \~chinese
/// \brief 读用户自定义数据。相机内部保留32768字节用于用户存储自定义数据(此功能针对本品牌相机，其它品牌相机无此功能)
/// \param handle [IN] 设备句柄
/// \param pBuffer [OUT] 数据缓冲的指针
/// \param pLength [IN] 期望读出的字节数 [OUT] 实际读出的字节数
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english 
/// \brief Read user-defined data; Inside the camera, there are 32768 bytes reserved for user to store private data (Only for our own camera has this function)
/// \param handle [IN] Device handle
/// \param pBuffer [OUT] Pointer of the data buffer
/// \param pLength [IN] Byte count read expected   [OUT] Byte count read in fact
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_ReadUserPrivateData(IN IMV_HANDLE handle, OUT void* pBuffer, IN_OUT unsigned int* pLength);

/// \~chinese
/// \brief 往相机串口寄存器写数据，每次写会清除掉上次的数据(此功能只支持包含串口功能的本品牌相机)
/// \param handle [IN] 设备句柄
/// \param pBuffer [IN] 数据缓冲的指针
/// \param pLength [IN] 期望写入的字节数 [OUT] 实际写入的字节数
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english 
/// \brief Write serial data to camera serial register, will erase the data writen before (Only for our own camera with serial port has this function)
/// \param handle [IN] Device handle
/// \param pBuffer [IN] Pointer of the data buffer
/// \param pLength [IN] Byte count written expected   [OUT] Byte count written in fact
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_WriteUARTData(IN IMV_HANDLE handle, IN void* pBuffer, IN_OUT unsigned int* pLength);

/// \~chinese
/// \brief 从相机串口寄存器读取串口数据(此功能只支持包含串口功能的本品牌相机 )
/// \param handle [IN] 设备句柄
/// \param pBuffer [OUT] 数据缓冲的指针
/// \param pLength [IN] 期望读出的字节数 [OUT] 实际读出的字节数
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english 
/// \brief Read serial data from camera serial register (Only for our own camera with serial port has this function)
/// \param handle [IN] Device handle
/// \param pBuffer [OUT] Pointer of the data buffer
/// \param pLength [IN] Byte count read expected   [OUT] Byte count read in fact
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_ReadUARTData(IN IMV_HANDLE handle, OUT void* pBuffer, IN_OUT unsigned int* pLength);

/// \~chinese
/// \brief 设备连接状态事件回调注册
/// \param handle [IN] 设备句柄
/// \param proc [IN] 设备连接状态事件回调函数
/// \param pUser [IN] 用户自定义数据, 可设为NULL
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 只支持一个回调函数,且设备关闭后，注册会失效，打开设备后需重新注册
/// \~english
/// \brief Register call back function of device connection status event.
/// \param handle [IN] Device handle
/// \param proc [IN] Call back function of device connection status event
/// \param pUser [IN] User defined data，It can be set to NULL
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// Only one call back function is supported.\n
/// Registration becomes invalid after the device is closed, , and need to re-register after the device is opened
IMV_API int IMV_CALL IMV_SubscribeConnectArg(IN IMV_HANDLE handle, IN IMV_ConnectCallBack proc, IN void* pUser);

/// \~chinese
/// \brief 参数更新事件回调注册
/// \param handle [IN] 设备句柄
/// \param proc [IN] 参数更新注册的事件回调函数
/// \param pUser [IN] 用户自定义数据, 可设为NULL
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 只支持一个回调函数,且设备关闭后，注册会失效，打开设备后需重新注册
/// \~english
/// \brief Register call back function of parameter update event.
/// \param handle [IN] Device handle
/// \param proc [IN] Call back function of parameter update event
/// \param pUser [IN] User defined data，It can be set to NULL
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// Only one call back function is supported.\n
/// Registration becomes invalid after the device is closed, , and need to re-register after the device is opened
IMV_API int IMV_CALL IMV_SubscribeParamUpdateArg(IN IMV_HANDLE handle, IN IMV_ParamUpdateCallBack proc, IN void* pUser);

/// \~chinese
/// \brief 流通道事件回调注册
/// \param handle [IN] 设备句柄
/// \param proc [IN] 流通道事件回调注册函数
/// \param pUser [IN] 用户自定义数据, 可设为NULL
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 只支持一个回调函数,且设备关闭后，注册会失效，打开设备后需重新注册
/// \~english
/// \brief Register call back function of stream channel event.
/// \param handle [IN] Device handle
/// \param proc [IN] Call back function of stream channel event
/// \param pUser [IN] User defined data，It can be set to NULL
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// Only one call back function is supported.\n
/// Registration becomes invalid after the device is closed, , and need to re-register after the device is opened
IMV_API int IMV_CALL IMV_SubscribeStreamArg(IN IMV_HANDLE handle, IN IMV_StreamCallBack proc, IN void* pUser);

/// \~chinese
/// \brief 消息通道事件回调注册
/// \param handle [IN] 设备句柄
/// \param proc [IN] 消息通道事件回调注册函数
/// \param pUser [IN] 用户自定义数据, 可设为NULL
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 只支持一个回调函数,且设备关闭后，注册会失效，打开设备后需重新注册
/// \~english
/// \brief Register call back function of message channel event.
/// \param handle [IN] Device handle
/// \param proc [IN] Call back function of message channel event
/// \param pUser [IN] User defined data，It can be set to NULL
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// Only one call back function is supported.\n
/// Registration becomes invalid after the device is closed, , and need to re-register after the device is opened
IMV_API int IMV_CALL IMV_SubscribeMsgChannelArg(IN IMV_HANDLE handle, IN IMV_MsgChannelCallBack proc, IN void* pUser);

/// \~chinese
/// \brief 设置帧数据缓存个数
/// \param handle [IN] 设备句柄
/// \param nSize [IN] 缓存数量
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 不能在拉流过程中设置
/// \~english
/// \brief Set frame buffer count
/// \param handle [IN] Device handle
/// \param nSize [IN] The buffer count
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// It can not be set during frame grabbing
IMV_API int IMV_CALL IMV_SetBufferCount(IN IMV_HANDLE handle, IN unsigned int nSize);

/// \~chinese
/// \brief 清除帧数据缓存
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Clear frame buffer
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_ClearFrameBuffer(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 设置驱动包间隔时间(MS),仅对Gige设备有效
/// \param handle [IN] 设备句柄
/// \param nTimeout [IN] 包间隔时间，单位是毫秒
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 触发模式尾包丢失重传机制
/// \~english
/// \brief Set packet timeout(MS), only for Gige device
/// \param handle [IN] Device handle
/// \param nTimeout [IN] Time out value, unit is MS
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// The resend mechanism of tail packet loss on trigger mode
IMV_API int IMV_CALL IMV_GIGE_SetInterPacketTimeout(IN IMV_HANDLE handle, IN unsigned int nTimeout);

/// \~chinese
/// \brief 设置单次重传最大包个数, 仅对GigE设备有效
/// \param handle [IN] 设备句柄
/// \param maxPacketNum [IN] 单次重传最大包个数
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// maxPacketNum为0时，该功能无效
/// \~english
/// \brief Set the single resend maximum packet number, only for Gige device
/// \param handle [IN] Device handle
/// \param maxPacketNum [IN] The value of single resend maximum packet number
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// Disable the function when maxPacketNum is 0
IMV_API int IMV_CALL IMV_GIGE_SetSingleResendMaxPacketNum(IN IMV_HANDLE handle, IN unsigned int maxPacketNum);

/// \~chinese
/// \brief 设置同一帧最大丢包的数量,仅对GigE设备有效
/// \param handle [IN] 设备句柄
/// \param maxLostPacketNum [IN] 最大丢包的数量
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// maxLostPacketNum为0时，该功能无效
/// \~english
/// \brief Set the maximum lost packet number, only for Gige device
/// \param handle [IN] Device handle
/// \param maxLostPacketNum [IN] The value of maximum lost packet number
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// Disable the function when maxLostPacketNum is 0
IMV_API int IMV_CALL IMV_GIGE_SetMaxLostPacketNum(IN IMV_HANDLE handle, IN unsigned int maxLostPacketNum);

/// \~chinese
/// \brief 设置U3V设备的传输数据块的数量和大小,仅对USB设备有效
/// \param handle [IN] 设备句柄
/// \param nNum	[IN] 传输数据块的数量(范围:5-256)
/// \param nSize [IN] 传输数据块的大小(范围:8-512, 单位:KByte)
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 1、传输数据块数量，范围5 - 256, 默认为64，高分辨率高帧率时可以适当增加该值；多台相机共同使用时，可以适当减小该值\n
/// 2、传输每个数据块大小，范围8 - 512, 默认为64，单位是KByte
/// \~english
/// \brief Set the number and size of urb transmitted, only for USB device
/// \param handle [IN] Device handle
/// \param nNum [IN] The number of urb transmitted(range:5-256)
/// \param nSize [IN] The size of urb transmitted（range:8-512, unit:KByte）
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// 1、The number of urb transmitted, the range is 5 - 256, and the default is 64. when high pixel and high frame rate can be appropriately increased.;
/// when multiple cameras are used together, the value can be appropriately reduced.\n
/// 2、The size of each urb transmitted, the range is 8 - 512, the default is 64, the unit is KByte.
IMV_API int IMV_CALL IMV_USB_SetUrbTransfer(IN IMV_HANDLE handle, IN unsigned int nNum, IN unsigned int nSize);

/// \~chinese
/// \brief 开始取流
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Start grabbing
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_StartGrabbing(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 开始取流
/// \param handle [IN] 设备句柄
/// \param maxImagesGrabbed [IN] 允许最多的取帧数，达到指定取帧数后停止取流，如果为0，表示忽略此参数连续取流(IMV_StartGrabbing默认0)
/// \param strategy [IN] 取流策略,(IMV_StartGrabbing默认使用grabStrartegySequential策略取流)
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Start grabbing
/// \param handle [IN] Device handle
/// \param maxImagesGrabbed [IN] Maximum images allowed to grab, once it reaches the limit then stop grabbing; 
/// If it is 0, then ignore this parameter and start grabbing continuously(default 0 in IMV_StartGrabbing)
/// \param strategy [IN] Image grabbing strategy; (Default grabStrartegySequential in IMV_StartGrabbing)
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_StartGrabbingEx(IN IMV_HANDLE handle, IN uint64_t maxImagesGrabbed, IN IMV_EGrabStrategy strategy);

/// \~chinese
/// \brief 判断设备是否正在取流
/// \param handle [IN] 设备句柄
/// \return 正在取流，返回true；不在取流，返回false
/// \~english
/// \brief Check whether device is grabbing or not
/// \param handle [IN] Device handle
/// \return Grabbing, return true. Not grabbing, return false 
IMV_API bool IMV_CALL IMV_IsGrabbing(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 停止取流
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Stop grabbing
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_StopGrabbing(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 注册帧数据回调函数(异步获取帧数据机制)
/// \param handle [IN] 设备句柄
/// \param proc [IN] 帧数据信息回调函数，建议不要在该函数中处理耗时的操作，否则会阻塞后续帧数据的实时性
/// \param pUser [IN] 用户自定义数据, 可设为NULL
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 该异步获取帧数据机制和同步获取帧数据机制(IMV_GetFrame)互斥，对于同一设备，系统中两者只能选其一\n
/// 只支持一个回调函数, 且设备关闭后，注册会失效，打开设备后需重新注册
/// \~english
/// \brief Register frame data callback function( asynchronous getting frame data mechanism);
/// \param handle [IN] Device handle
/// \param proc [IN] Frame data information callback function; It is advised to not put time-cosuming operation in this function, 
/// otherwise it will block follow-up data frames and affect real time performance
/// \param pUser [IN] User defined data，It can be set to NULL
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// This asynchronous getting frame data mechanism and synchronous getting frame data mechanism(IMV_GetFrame) are mutually exclusive,\n
/// only one method can be choosed between these two in system for the same device.\n
/// Only one call back function is supported.\n
/// Registration becomes invalid after the device is closed, , and need to re-register after the device is opened
IMV_API int IMV_CALL IMV_AttachGrabbing(IN IMV_HANDLE handle, IN IMV_FrameCallBack proc, IN void* pUser);

/// \~chinese
/// \brief 获取一帧图像(同步获取帧数据机制)
/// \param handle [IN] 设备句柄
/// \param pFrame [OUT] 帧数据信息
/// \param timeoutMS [IN] 获取一帧图像的超时时间,INFINITE时表示无限等待,直到收到一帧数据或者停止取流。单位是毫秒
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 该接口不支持多线程调用。\n
/// 该同步获取帧机制和异步获取帧机制(IMV_AttachGrabbing)互斥,对于同一设备，系统中两者只能选其一。\n
/// 使用内部缓存获取图像，需要IMV_ReleaseFrame进行释放图像缓存。
/// \~english
/// \brief Get a frame image(synchronous getting frame data mechanism)
/// \param handle [IN] Device handle
/// \param pFrame [OUT] Frame data information
/// \param timeoutMS [IN] The time out of getting one image, INFINITE means infinite wait until the one frame data is returned or stop grabbing.unit is MS
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// This interface does not support multi-threading.\n
/// This synchronous getting frame data mechanism and asynchronous getting frame data mechanism(IMV_AttachGrabbing) are mutually exclusive,\n
/// only one method can be chose between these two in system for the same device.\n
/// Use internal cache to get image, need to release image buffer by IMV_ReleaseFrame
IMV_API int IMV_CALL IMV_GetFrame(IN IMV_HANDLE handle, OUT IMV_Frame* pFrame, IN unsigned int timeoutMS);

/// \~chinese
/// \brief 释放图像缓存
/// \param handle [IN] 设备句柄
/// \param pFrame [IN] 帧数据信息
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Free image buffer
/// \param handle [IN] Device handle
/// \param pFrame [IN] Frame image data information
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_ReleaseFrame(IN IMV_HANDLE handle, IN IMV_Frame* pFrame);

/// \~chinese
/// \brief 帧数据深拷贝克隆
/// \param handle [IN] 设备句柄
/// \param pFrame [IN] 克隆源帧数据信息
/// \param pCloneFrame [OUT] 新的帧数据信息
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 使用IMV_ReleaseFrame进行释放图像缓存。
/// \~english
/// \brief Frame data deep clone
/// \param handle [IN] Device handle
/// \param pFrame [IN] Frame data information of clone source
/// \param pCloneFrame [OUT] New frame data information
/// \return Success, return IMV_OK. Failure, return error code 
/// \remarks
/// Use IMV_ReleaseFrame to free image buffer

IMV_API int IMV_CALL IMV_CloneFrame(IN IMV_HANDLE handle, IN IMV_Frame* pFrame, OUT IMV_Frame* pCloneFrame);

/// \~chinese
/// \brief 获取Chunk数据(仅对GigE/Usb相机有效)
/// \param handle [IN] 设备句柄
/// \param pFrame [IN] 帧数据信息
/// \param index [IN] 索引ID
/// \param pChunkDataInfo [OUT] Chunk数据信息
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get chunk data(Only GigE/Usb Camera)
/// \param handle [IN] Device handle
/// \param pFrame [IN] Frame data information
/// \param index [IN] index ID
/// \param pChunkDataInfo [OUT] Chunk data infomation
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetChunkDataByIndex(IN IMV_HANDLE handle, IN IMV_Frame* pFrame, IN unsigned int index, OUT IMV_ChunkDataInfo *pChunkDataInfo);

/// \~chinese
/// \brief 获取流统计信息(IMV_StartGrabbing / IMV_StartGrabbing执行后调用)
/// \param handle [IN] 设备句柄
/// \param pStreamStatsInfo [OUT] 流统计信息数据
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get stream statistics infomation(Used after excuting IMV_StartGrabbing / IMV_StartGrabbing)
/// \param handle [IN] Device handle
/// \param pStreamStatsInfo [OUT] Stream statistics infomation
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetStatisticsInfo(IN IMV_HANDLE handle, OUT IMV_StreamStatisticsInfo* pStreamStatsInfo);

/// \~chinese
/// \brief 重置流统计信息(IMV_StartGrabbing / IMV_StartGrabbing执行后调用)
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Reset stream statistics infomation(Used after excuting IMV_StartGrabbing / IMV_StartGrabbing)
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_ResetStatisticsInfo(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 判断属性是否可用
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \return 可用，返回true；不可用，返回false
/// \~english
/// \brief Check the property is available or not
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \return Available, return true. Not available, return false 
IMV_API bool IMV_CALL IMV_FeatureIsAvailable(IN IMV_HANDLE handle, IN const char* pFeatureName);

/// \~chinese
/// \brief 判断属性是否可读
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \return 可读，返回true；不可读，返回false
/// \~english
/// \brief Check the property is readable or not
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \return Readable, return true. Not readable, return false 
IMV_API bool IMV_CALL IMV_FeatureIsReadable(IN IMV_HANDLE handle, IN const char* pFeatureName);

/// \~chinese
/// \brief 判断属性是否可写
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \return 可写，返回true；不可写，返回false
/// \~english
/// \brief Check the property is writeable or not
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \return Writeable, return true. Not writeable, return false 
IMV_API bool IMV_CALL IMV_FeatureIsWriteable(IN IMV_HANDLE handle, IN const char* pFeatureName);

/// \~chinese
/// \brief 判断属性是否可流
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \return 可流，返回true；不可流，返回false
/// \~english
/// \brief Check the property is streamable or not
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \return Streamable, return true. Not streamable, return false 
IMV_API bool IMV_CALL IMV_FeatureIsStreamable(IN IMV_HANDLE handle, IN const char* pFeatureName);

/// \~chinese
/// \brief 判断属性是否有效
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \return 有效，返回true；无效，返回false
/// \~english
/// \brief Check the property is valid or not
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \return Valid, return true. Invalid, return false 
IMV_API bool IMV_CALL IMV_FeatureIsValid(IN IMV_HANDLE handle, IN const char* pFeatureName);

/// \~chinese
/// \brief 获取属性类型
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pPropertyType [OUT] 属性类型
/// \return 获取成功，返回true；获取失败，返回false
/// \~english
/// \brief get property type
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \return get success, return true. get failed, return false 
IMV_API bool IMV_CALL IMV_GetFeatureType(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT IMV_EFeatureType* pPropertyType);

/// \~chinese
/// \brief 获取整型属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pIntValue [OUT] 整型属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get integer property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pIntValue [OUT] Integer property value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetIntFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);

/// \~chinese
/// \brief 获取整型属性可设的最小值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pIntValue [OUT] 整型属性可设的最小值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get the integer property settable minimum value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pIntValue [OUT] Integer property settable minimum value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetIntFeatureMin(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);

/// \~chinese
/// \brief 获取整型属性可设的最大值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pIntValue [OUT] 整型属性可设的最大值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get the integer property settable maximum value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pIntValue [OUT] Integer property settable maximum value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetIntFeatureMax(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);

/// \~chinese
/// \brief 获取整型属性步长
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pIntValue [OUT] 整型属性步长
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get integer property increment
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pIntValue [OUT] Integer property increment
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetIntFeatureInc(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT int64_t* pIntValue);

/// \~chinese
/// \brief 设置整型属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param intValue [IN] 待设置的整型属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Set integer property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param intValue [IN] Integer property value to be set 
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_SetIntFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN int64_t intValue);

/// \~chinese
/// \brief 获取浮点属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pDoubleValue [OUT] 浮点属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get double property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pDoubleValue [OUT] Double property value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetDoubleFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT double* pDoubleValue);

/// \~chinese
/// \brief 获取浮点属性可设的最小值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pDoubleValue [OUT] 浮点属性可设的最小值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get the double property settable minimum value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pDoubleValue [OUT] Double property settable minimum value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetDoubleFeatureMin(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT double* pDoubleValue);

/// \~chinese
/// \brief 获取浮点属性可设的最大值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pDoubleValue [OUT] 浮点属性可设的最大值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get the double property settable maximum value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pDoubleValue [OUT] Double property settable maximum value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetDoubleFeatureMax(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT double* pDoubleValue);

/// \~chinese
/// \brief 设置浮点属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param doubleValue [IN] 待设置的浮点属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Set double property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param doubleValue [IN] Double property value to be set 
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_SetDoubleFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN double doubleValue);

/// \~chinese
/// \brief 获取布尔属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pBoolValue [OUT] 布尔属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get boolean property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pBoolValue [OUT] Boolean property value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetBoolFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT bool* pBoolValue);

/// \~chinese
/// \brief 设置布尔属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param boolValue [IN] 待设置的布尔属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Set boolean property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param boolValue [IN] Boolean property value to be set 
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_SetBoolFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN bool boolValue);

/// \~chinese
/// \brief 获取枚举属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pEnumValue [OUT] 枚举属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get enumeration property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pEnumValue [OUT] Enumeration property value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetEnumFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT uint64_t* pEnumValue);

/// \~chinese
/// \brief 设置枚举属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param enumValue [IN] 待设置的枚举属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Set enumeration property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param enumValue [IN] Enumeration property value to be set 
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_SetEnumFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN uint64_t enumValue);

/// \~chinese
/// \brief 获取枚举属性symbol值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pEnumSymbol [OUT] 枚举属性symbol值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get enumeration property symbol value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pEnumSymbol [OUT] Enumeration property symbol value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetEnumFeatureSymbol(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT IMV_String* pEnumSymbol);

/// \~chinese
/// \brief 设置枚举属性symbol值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pEnumSymbol [IN] 待设置的枚举属性symbol值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Set enumeration property symbol value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pEnumSymbol [IN] Enumeration property symbol value to be set 
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_SetEnumFeatureSymbol(IN IMV_HANDLE handle, IN const char* pFeatureName, IN const char* pEnumSymbol);

/// \~chinese
/// \brief 获取枚举属性的可设枚举值的个数
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pEntryNum [OUT] 枚举属性的可设枚举值的个数
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get the number of enumeration property settable enumeration
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pEntryNum [OUT] The number of enumeration property settable enumeration value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetEnumFeatureEntryNum(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT unsigned int* pEntryNum);

/// \~chinese
/// \brief 获取枚举属性的可设枚举值列表
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pEnumEntryList [OUT] 枚举属性的可设枚举值列表
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get settable enumeration value list of enumeration property 
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pEnumEntryList [OUT] Settable enumeration value list of enumeration property 
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetEnumFeatureEntrys(IN IMV_HANDLE handle, IN const char* pFeatureName, IN_OUT IMV_EnumEntryList* pEnumEntryList);

/// \~chinese
/// \brief 获取字符串属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pStringValue [OUT] 字符串属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Get string property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pStringValue [OUT] String property value
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_GetStringFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, OUT IMV_String* pStringValue);

/// \~chinese
/// \brief 设置字符串属性值
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \param pStringValue [IN] 待设置的字符串属性值
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Set string property value
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \param pStringValue [IN] String property value to be set
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_SetStringFeatureValue(IN IMV_HANDLE handle, IN const char* pFeatureName, IN const char* pStringValue);

/// \~chinese
/// \brief 执行命令属性
/// \param handle [IN] 设备句柄
/// \param pFeatureName [IN] 属性名
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Execute command property
/// \param handle [IN] Device handle
/// \param pFeatureName [IN] Feature name
/// \return Success, return IMV_OK. Failure, return error code 
IMV_API int IMV_CALL IMV_ExecuteCommandFeature(IN IMV_HANDLE handle, IN const char* pFeatureName);

/// \~chinese
/// \brief 像素格式转换
/// \param handle [IN] 设备句柄
/// \param pstPixelConvertParam [IN][OUT] 像素格式转换参数结构体
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 只支持转化成目标像素格式gvspPixelRGB8 / gvspPixelBGR8 / gvspPixelMono8 / gvspPixelBGRA8\n
/// 通过该接口将原始图像数据转换成用户所需的像素格式并存放在调用者指定内存中。\n
/// 像素格式为YUV411Packed的时，图像宽须能被4整除\n
/// 像素格式为YUV422Packed的时，图像宽须能被2整除\n
/// 像素格式为YUYVPacked的时，图像宽须能被2整除\n
/// 转换后的图像:数据存储是从最上面第一行开始的，这个是相机数据的默认存储方向
/// \~english
/// \brief Pixel format conversion
/// \param handle [IN] Device handle
/// \param pstPixelConvertParam [IN][OUT] Convert Pixel Type parameter structure
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// Only support converting to destination pixel format of gvspPixelRGB8 / gvspPixelBGR8 / gvspPixelMono8 / gvspPixelBGRA8\n
/// This API is used to transform the collected original data to pixel format and save to specified memory by caller.\n
/// pixelFormat:YUV411Packed, the image width is divisible by 4\n
/// pixelFormat : YUV422Packed, the image width is divisible by 2\n
/// pixelFormat : YUYVPacked，the image width is divisible by 2\n
/// converted image：The first row of the image is located at the start of the image buffer.This is the default for image taken by a camera.
IMV_API int IMV_CALL IMV_PixelConvert(IN IMV_HANDLE handle, IN_OUT IMV_PixelConvertParam* pstPixelConvertParam);

/// \~chinese
/// \brief 打开录像
/// \param handle [IN] 设备句柄
/// \param pstRecordParam [IN] 录像参数结构体
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Open record
/// \param handle [IN] Device handle
/// \param pstRecordParam [IN] Record param structure
/// \return Success, return IMV_OK. Failure, return error code
IMV_API int IMV_CALL IMV_OpenRecord(IN IMV_HANDLE handle, IN IMV_RecordParam *pstRecordParam);

/// \~chinese
/// \brief 录制一帧图像
/// \param handle [IN] 设备句柄
/// \param pstRecordFrameInfoParam [IN] 录像用帧信息结构体
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Record one frame
/// \param handle [IN] Device handle
/// \param pstRecordFrameInfoParam [IN] Frame information for recording structure
/// \return Success, return IMV_OK. Failure, return error code
IMV_API int IMV_CALL IMV_InputOneFrame(IN IMV_HANDLE handle, IN IMV_RecordFrameInfoParam *pstRecordFrameInfoParam);

/// \~chinese
/// \brief 关闭录像
/// \param handle [IN] 设备句柄
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \~english
/// \brief Close record
/// \param handle [IN] Device handle
/// \return Success, return IMV_OK. Failure, return error code
IMV_API int IMV_CALL IMV_CloseRecord(IN IMV_HANDLE handle);

/// \~chinese
/// \brief 图像翻转
/// \param handle [IN] 设备句柄
/// \param pstFlipImageParam [IN][OUT] 图像翻转参数结构体
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 只支持像素格式gvspPixelRGB8 / gvspPixelBGR8 / gvspPixelMono8的图像的垂直和水平翻转。\n
/// 通过该接口将原始图像数据翻转后并存放在调用者指定内存中。
/// \~english
/// \brief Flip image
/// \param handle [IN] Device handle
/// \param pstFlipImageParam [IN][OUT] Flip image parameter structure
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// Only support vertical and horizontal flip of image data with gvspPixelRGB8 / gvspPixelBGR8 / gvspPixelMono8 pixel format.\n
/// This API is used to flip original data and save to specified memory by caller.
IMV_API int IMV_CALL IMV_FlipImage(IN IMV_HANDLE handle, IN_OUT IMV_FlipImageParam* pstFlipImageParam);

/// \~chinese
/// \brief 图像顺时针旋转
/// \param handle [IN] 设备句柄
/// \param pstRotateImageParam [IN][OUT] 图像旋转参数结构体
/// \return 成功，返回IMV_OK；错误，返回错误码
/// \remarks
/// 只支持gvspPixelRGB8 / gvspPixelBGR8 / gvspPixelMono8格式数据的90/180/270度顺时针旋转。\n
/// 通过该接口将原始图像数据旋转后并存放在调用者指定内存中。
/// \~english
/// \brief Rotate image clockwise
/// \param handle [IN] Device handle
/// \param pstRotateImageParam [IN][OUT] Rotate image parameter structure
/// \return Success, return IMV_OK. Failure, return error code
/// \remarks
/// Only support 90/180/270 clockwise rotation of data in the gvspPixelRGB8 / gvspPixelBGR8 / gvspPixelMono8 format.\n
/// This API is used to rotation original data and save to specified memory by caller.
IMV_API int IMV_CALL IMV_RotateImage(IN IMV_HANDLE handle, IN_OUT IMV_RotateImageParam* pstRotateImageParam);

#ifdef __cplusplus
}
#endif 

#endif // __IMV_API_H__