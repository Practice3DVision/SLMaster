pragma Singleton

import QtQuick 2.15

QtObject {
    property string settings
    property string locale
    property string about
    property string system_mode
    property string dark_mode
    property string light_mode
    property string exit
    property string exit_confirm
    property string minimum
    property string frendly_tip
    property string hide_tip
    property string cancel
    property string device
    property string calibration
    property string img_display
    property string scan_mode
    property string scan
    property string post_process
    property string post_process_output
    property string v_sync
    property string use_system_bar
    property string fits_app_bar
    property string software_render
    property string check_restart
    property string confirm
    property string author
    property string evansLiu
    property string about_info
    property string wechat
    property string online
    property string offline
    property string monocular_sl_camera
    property string binocular_sl_camera
    property string triple_sl_camera
    property string connect
    property string stripe_encoding
    property string pixel_depth
    property string one_bit
    property string eight_bit
    property string camera
    property string stripe_type
    property string sine_complementary_gray_code
    property string interzone_sinus_four_grayscale
    property string encode
    property string save
    property string defocus_encoding
    property string opwm
    property string error_diffusion_method
    property string img_width
    property string img_height
    property string cycles
    property string shift_time
    property string stripe_img
    property string burn
    property string previous_text
    property string next_text
    property string direction
    property string honrizon
    property string vertical
    property string connect_sucess
    property string connect_failed
    property string disconnect_sucess
    property string disconnect_failed
    property string connect_ing
    property string burn_ing
    property string burn_finished
    property string disconnect
    property string disconnect_ing
    property string save_file
    property string please_choose_save_folder
    property string disable
    property string binary
    property string one_bit_disable_defocus_warning
    property string calibration_type_select
    property string camera_offline_calibration
    property string camera_offline_calibration_tip
    property string camera_online_calibration
    property string camera_online_calibration_tip
    property string projector_online_calibration
    property string projector_online_calibration_tip
    property string img_browser
    property string please_select_left_folder
    property string please_select_right_folder
    property string please_select_color_folder
    property string select_left_folder
    property string select_right_folder
    property string select_color_folder
    property string left_camera
    property string right_camera
    property string color_camera
    property string remove
    property string target_type
    property string row_feature_points_num
    property string col_feature_points_num
    property string features_distance
    property string inner_circle_inner_radius
    property string inner_circle_externer_radius
    property string externer_circle_inner_radius
    property string externer_circle_externer_radius
    property string chess_board
    property string blob
    property string concentric_circle
    property string single_calibration
    property string stereo_calibration
    property string triple_calibration
    property string calibration_ing
    property string calibration_error
    property string save_sucess
    property string error_point_distribution
    property string img_error_distribution
    property string result_display
    property string display_rectify_img
    property string export_as_left_color_calibration
    property string capture
    property string set_calibration_params
    property string cam_img
    property string phase_h
    property string phase_v
    property string projector_img
    property string honrizon_pitch
    property string vertical_pitch
    property string calibrate_intrincis
    property string calibrate_extrincis
    property string capture_sucess
    property string capture_failed
    property string projector_camera
    property string static_scan_mode
    property string dynamic_scan_mode
    property string offline_scan_mode
    property string static_scan_mode_tip
    property string dynamic_scan_mode_tip
    property string offline_scan_mode_tip
    property string scan_mode_selcet
    property string start_scan
    property string light_strength
    property string exposure_time
    property string filter_threshod
    property string pattern_type
    property string enable_gpu
    property string texture
    property string sinus_comple_gray_code
    property string three_frequency_heterodyne
    property string reserved
    property string please_select
    property string select_finished
    property string please_select_start_point
    property string please_select_point_to_see_info
    property string multi_view_stereo_geometry
    property string sine_shift_gray_code
    property string find_features_error
    property string pure_scan_mode
    property string pure_scan_desc
    property string turret_scan_mode
    property string turret_scan_desc
    property string configure_static_scan_algorithm
    property string device_function_select
    property string pre_exposure_time
    property string aft_exposure_time
    property string project_once
    property string project_continue
    property string project_step
    property string stop_project
    property string pause_project
    property string project_ten_line
    property string clear_pre_stripes
    property string keep_add_stripes
    property string operation_dialog
    property string execute
    property string nodes
    property string inoutput
    property string filters
    property string registration
    property string sample_consensus
    property string segmentation
    property string surface
    property string features
    property string cloudInputNode
    property string cloudOutputNode
    property string passThroughFilterNode
    property string cloudInputMode
    property string selectCloudFile
    property string fromFile
    property string fromCamera
    property string enableX
    property string minX
    property string maxX
    property string enableY
    property string minY
    property string maxY
    property string enableZ
    property string minZ
    property string maxZ
    property string staticRemovel
    property string meanK
    property string stdThresh
    property string group
    property string greedyProjectionTriangulation
    property string kSearch
    property string multiplier
    property string maxNearestNumber
    property string searchRadius
    property string minimumAngle
    property string maximumAngle
    property string maximumSurfaceAngle
    property string normalConsistency
    property string meshOutputNode
    property string poisson
    property string minDepth
    property string maxDepth
    property string scale
    property string solverDivide
    property string isoDivide
    property string minSamplePoints
    property string degree
    property string confidence
    property string manifold
    property string read_params
    property string export_epiline
    property string model_type
    property string method_type
    property string distance_threshold
    property string sac_RANSAC
    property string sac_LMEDS
    property string sac_MSAC
    property string sac_RRANSAC
    property string sac_RMSAC
    property string sac_MLESAC
    property string sac_PROSAC
    property string sacMODEL_PLANE
    property string sacMODEL_LINE
    property string sacMODEL_CIRCLE2D
    property string sacMODEL_CIRCLE3D
    property string sacMODEL_SPHERE
    property string sacMODEL_CYLINDER
    property string sacMODEL_CONE
    property string sacMODEL_TORUS
    property string sacMODEL_PARALLEL_LINE
    property string sacMODEL_PERPENDICULAR_PLANE
    property string sacMODEL_PARALLEL_LINES
    property string sacMODEL_NORMAL_PLANE
    property string sacMODEL_NORMAL_SPHERE
    property string sacMODEL_REGISTRATION
    property string sacMODEL_REGISTRATION_2D
    property string sacMODEL_PARALLEL_PLANE
    property string sacMODEL_NORMAL_PARALLEL_LINES
    property string sacMODEL_STICK
    property string sacMODEL_ELLIPSE3D
    property string sac_segment
    property string split_output_node
    property string split_port_id
    property string intersection_line_node
    property string three_line_intersection_node
    property string actor_output_node
    property string width
    property string height
    property string length
    property string sequenceInverse
    property string clip_width
    property string clip_height
    property string horizonVerticalInv
    property string useCurrentFeaturePoints

    function zh() {
        settings = "设置";
        locale = "语言环境";
        about = "关于";
        system_mode = "跟随系统";
        light_mode = "明亮模式";
        dark_mode = "夜间模式";
        v_sync = "垂直同步";
        exit = "退出";
        exit_confirm = "是否退出程序？请确认结果已保存。";
        minimum = "最小化";
        frendly_tip = "友情提示";
        hide_tip = "SLMasterGui已隐藏至托盘，点击托盘可再次激活窗口";
        cancel = "取消";
        device = "设备";
        calibration = "标定";
        scan_mode = "扫描模式";
        scan = "扫描";
        post_process = "后处理";
        post_process_output = "后处理结果";
        use_system_bar = "使用系统窗口";
        fits_app_bar = "自动调整应用程序托盘";
        software_render = "软件渲染";
        check_restart = "此操作需要重启应用程序，请确定重新打开应用程序?";
        confirm = "确认";
        author = "作者";
        evansLiu = "Evans Liu";
        about_info = " 该软件为机构光相机软件，\n \n 如需GPU加速请从源码编译! \n \n 请不要将本软件用于包括销售、推广等任何非经许可的商业用途! \n \n 关注公众号：实战3D视觉，更多技术内容与你分享。";
        wechat = " 付费咨询与项目合作请加微信：";
        online = "在线";
        offline = "离线";
        monocular_sl_camera = "单目结构光相机";
        binocular_sl_camera = "双目结构光相机";
        triple_sl_camera = "三目结构光相机";
        connect = "连接";
        stripe_encoding = "条纹编码";
        pixel_depth = "像素深度"
        one_bit = "一位";
        eight_bit = "八位";
        camera = "相机";
        stripe_type = "条纹类型";
        sine_complementary_gray_code = "正弦互补格雷码";
        sine_shift_gray_code = "正弦移位格雷码";
        interzone_sinus_four_grayscale = "四灰度编码广义分区间相位展开";
        encode = "编码";
        save = "保存";
        defocus_encoding = "离焦编码";
        opwm = "最佳脉冲宽度调制";
        error_diffusion_method = "二维误差扩散法";
        img_width = "图像宽度";
        img_height = "图像高度";
        cycles = "周期数";
        shift_time = "相移次数";
        stripe_img = "条纹图片";
        burn = "烧录";
        previous_text = "<上一页";
        next_text = "下一页>";
        direction = "方向";
        honrizon = "水平";
        vertical = "垂直";
        connect_sucess = "连接成功!";
        connect_failed = "连接失败!";
        disconnect_sucess = "断开连接成功!";
        disconnect_failed = "断开连接失败!";
        connect_ing = "连接中...";
        burn_ing = "条纹烧录中...";
        burn_finished = "条纹烧录完成!";
        disconnect = "断开连接";
        disconnect_ing = "断开连接中...";
        save_file = "保存文件";
        save_sucess = "保存成功";
        please_choose_save_folder = "请选择文件夹";
        disable = "不使用";
        binary = "二进制";
        one_bit_disable_defocus_warning = "1 bit条纹编码必须选择离焦方法进行编码";
        calibration_type_select = "标定类型选择";
        camera_offline_calibration = "离线相机标定";
        camera_offline_calibration_tip = "离线相机标定通过离线采集的标定板图片，提取特征点并进行单目、双目标定。";
        camera_online_calibration = "在线相机标定";
        camera_online_calibration_tip = "在线相机标定通过相机在线采集标定板图片，提取特征点并进行单目、双目标定。";
        projector_online_calibration = "在线投影仪标定";
        projector_online_calibration_tip = "在线投影仪标定通过辅助相机逆投影法采集投影仪投出的正弦格雷码图案并进行解相，以此使得相机帮助投影仪进行标定板图片的采集。该程序将生成八参数标定文件以及投影仪内参。";
        img_browser = "图片浏览器";
        please_select_left_folder = "请选择左相机捕获的标定板图片所在文件夹";
        please_select_right_folder = "请选择右相机捕获的标定板图片所在文件夹";
        please_select_color_folder = "请选择彩色相机捕获的标定板图片所在文件夹";
        select_left_folder = "选择左相机文件夹";
        select_right_folder = "选择右相机文件夹";
        select_color_folder = "选择彩色相机文件夹";
        left_camera = "左相机";
        right_camera = "右相机";
        color_camera = "彩色相机";
        remove = "移除";
        target_type = "标定靶标类型";
        row_feature_points_num = "行特征点数";
        col_feature_points_num = "列特征点数";
        features_distance = "特征点距离";
        inner_circle_inner_radius = "内圆环内圆半径";
        inner_circle_externer_radius = "内圆环外圆半径";
        externer_circle_inner_radius = "外圆环内圆半径";
        externer_circle_externer_radius = "外圆环外圆半径";
        chess_board = "棋盘格";
        blob = "圆点";
        concentric_circle = "同心圆环";
        single_calibration = "单目标定";
        stereo_calibration = "双目标定";
        triple_calibration = "三目标定";
        calibration_ing = "标定中";
        calibration_error = "标定误差";
        error_point_distribution = "当前图片特征点误差分布图";
        img_error_distribution = "重投影均方根误差: "
        result_display = "标定结果显示";
        display_rectify_img = "极线校正实例";
        export_as_left_color_calibration = "作为左相机-彩色相机标定参数导出";
        capture = "捕获";
        set_calibration_params = "设置标定参数";
        cam_img = "相机图片";
        phase_h = "水平相位";
        phase_v = "垂直相位";
        projector_img = "投影仪图片";
        honrizon_pitch = "水平节距";
        vertical_pitch = "垂直节距";
        calibrate_intrincis = "标定内参";
        calibrate_extrincis = "标定外参";
        capture_sucess = "捕获成功";
        capture_failed = "捕获失败,请检查是否无法成功提取特征点";
        projector_camera = "投影仪";
        static_scan_mode = "静态扫描模式";
        dynamic_scan_mode = "动态扫描模式";
        offline_scan_mode = "离线扫描模式";
        static_scan_mode_tip = "静态扫描模式可实现更准确、更易于操作的数据处理。";
        dynamic_scan_mode_tip = "动态扫描模式可以以系统最高的3D重建速率运行，并具有3D重建动态物体的能力。";
        offline_scan_mode_tip = "离线扫描模式可以方便处理离线数据，此外，也更能方便开发者开发它们自己的算法。";
        scan_mode_selcet = "扫描模式选择：";
        start_scan = "开始扫描";
        light_strength = "投影光强";
        exposure_time = "曝光时间";
        filter_threshod = "调制度过滤阈值";
        pattern_type = "图案类型";
        enable_gpu = "GPU加速";
        texture = "纹理";
        sinus_comple_gray_code = "正弦互补格雷码";
        three_frequency_heterodyne = "三频外差";
        reserved = "保留";
        please_select = "请通过右键点击屏幕以绘制包围盒！";
        select_finished = "包围盒绘制完成！";
        please_select_start_point = "请通过左键点击屏幕以开始包围盒绘制!";
        please_select_point_to_see_info = "请点击点以查询具体信息";
        multi_view_stereo_geometry = "多视立体几何约束";
        find_features_error = "图片无法找到特征点，请检查参数和图片!图片路径： %1";
        pure_scan_mode = "纯净扫描";
        turret_scan_mode = "转台拼接配置";
        configure_static_scan_algorithm = "配置静态扫描算法";
        pure_scan_desc = "不使用任何辅助设备或者辅助算法";
        turret_scan_desc = "使用转台辅助设备对任意物体进行多次旋转扫描，以达到拼接效果";
        device_function_select = "功能选择";
        pre_exposure_time = "前置曝光时间";
        aft_exposure_time = "后置曝光时间";
        project_once = "投影一次";
        project_continue = "连续投影";
        project_step = "步进投影";
        stop_project = "停止投影";
        pause_project = "暂停投影";
        project_ten_line = "十字线";
        clear_pre_stripes = "清除前序编码条纹";
        keep_add_stripes = "保留前序编码条纹";
        operation_dialog = "操作对话框";
        execute = "执行";
        nodes = "节点";
        inoutput = "输入/输出";
        filters = "过滤器";
        registration = "注册对齐";
        sample_consensus = "抽样拟合";
        segmentation = "分割";
        surface = "曲面化";
        features = "特征";
        cloudInputNode = "点云输入节点";
        cloudOutputNode = "点云输出节点";
        passThroughFilterNode = "直通滤波器";
        cloudInputMode = "点云输入模式";
        selectCloudFile = "选择点云文件";
        fromFile = "离线点云文件";
        fromCamera = "在线相机点云";
        enableX = "X方向滤波";
        minX = "X方向最小值";
        maxX = "X方向最大值";
        enableY = "Y方向滤波";
        minY = "Y方向最小值";
        maxY = "Y方向最大值";
        enableZ = "Z方向滤波";
        minZ = "Z方向最小值";
        maxZ = "Z方向最大值";
        staticRemovel = "统计学滤波器";
        meanK = "最近邻近点数量"
        stdThresh = "标准偏差系数";
        group = "组";
        greedyProjectionTriangulation = "贪婪三角化";
        kSearch = "二叉树邻近点搜索数量";
        multiplier = "加权因子";
        maxNearestNumber = "最大邻域点个数";
        searchRadius = "搜索半径";
        minimumAngle = "三角形面体最小角";
        maximumAngle = "三角形面体最大角";
        maximumSurfaceAngle = "临近点与样本点法线最大偏离角度";
        normalConsistency = "法线朝向一致";
        meshOutputNode = "面元输出节点";
        poisson = "泊松重建节点";
        minDepth = "最小深度";
        maxDepth = "最大深度";
        scale = "重构与样本的立方体直径比率";
        solverDivide = "线性迭代方法的深度";
        isoDivide = "提取ISO等值面的深度";
        minSamplePoints = "八叉树节点中的样本点数量";
        degree = "精细度";
        confidence = "是否使用法向量作为置信信息";
        manifold = "是否添加多边形的重心";
        img_display = "图片展示";
        read_params = "读取参数";
        export_epiline = "输出极线系数矩阵";
        model_type = "模型类型";
        method_type = "方法类型";
        distance_threshold = "距离阈值";
        sac_RANSAC = "SAC_RANSAC";
        sac_LMEDS = "SAC_LMEDS";
        sac_MSAC = "SAC_MSAC";
        sac_RRANSAC = "SAC_RRANSAC";
        sac_RMSAC = "SAC_RMSAC";
        sac_MLESAC = "SAC_MLESAC";
        sac_PROSAC = "SAC_PROSAC";
        sacMODEL_PLANE = "SACMODEL_PLANE";
        sacMODEL_LINE = "SACMODEL_LINE";
        sacMODEL_CIRCLE2D = "SACMODEL_CIRCLE2D";
        sacMODEL_CIRCLE3D = "SACMODEL_CIRCLE3D";
        sacMODEL_SPHERE = "SACMODEL_SPHERE";
        sacMODEL_CYLINDER = "SACMODEL_CYLINDER";
        sacMODEL_CONE = "SACMODEL_CONE";
        sacMODEL_TORUS = "SACMODEL_TORUS";
        sacMODEL_PARALLEL_LINE = "SACMODEL_PARALLEL_LINE";
        sacMODEL_PERPENDICULAR_PLANE = "SACMODEL_PERPENDICULAR_PLANE";
        sacMODEL_PARALLEL_LINES = "SACMODEL_PARALLEL_LINES";
        sacMODEL_NORMAL_PLANE = "SACMODEL_NORMAL_PLANE";
        sacMODEL_NORMAL_SPHERE = "SACMODEL_NORMAL_SPHERE";
        sacMODEL_REGISTRATION = "SACMODEL_REGISTRATION";
        sacMODEL_REGISTRATION_2D = "SACMODEL_REGISTRATION_2D";
        sacMODEL_PARALLEL_PLANE = "SACMODEL_PARALLEL_PLANE";
        sacMODEL_NORMAL_PARALLEL_LINES = "SACMODEL_NORMAL_PARALLEL_LINES";
        sacMODEL_STICK = "SACMODEL_STICK";
        sacMODEL_ELLIPSE3D = "SACMODEL_ELLIPSE3D";
        sac_segment = "采样一致性分割";
        split_output_node = "输出分离节点";
        split_port_id = "端口ID";
        intersection_line_node = "面与面交线节点";
        three_line_intersection_node = "三线相交节点";
        actor_output_node = "渲染组件输出节点";
        width = "宽度";
        height = "高度";
        length = "长度";
        sequenceInverse = "特征点顺序翻转";
        horizonVerticalInv = "水平/垂直特征点顺序翻转";
        clip_width = "裁剪宽度";
        clip_height = "裁剪高度";
        useCurrentFeaturePoints = "使用当前的特征点";
    }

    function en() {
        settings = "Settings";
        locale = "Locale";
        about = "About";
        system_mode = "System Mode";
        light_mode = "Light Mode";
        dark_mode = "Dark Mode";
        exit = "Exit";
        exit_confirm = "Do you want to quit the program? Please confirm that the result has been saved.";
        minimum = "Minimumize";
        frendly_tip ="Frendly Tip";
        hide_tip = "SLMasterGui is hidden from the tray, click on the tray to activate the window again ...";
        cancel = "Cancel";
        device = "Device";
        calibration = "Calibration";
        scan_mode = "Scan Mode";
        scan = "Scan";
        post_process = "Post Process";
        post_process_output = "Post Process Output";
        v_sync = "V-SYNC";
        use_system_bar = "Use System Bar";
        fits_app_bar = "Fits App Bar";
        software_render = "Software Render";
        check_restart = "This operation requires restarting the application. Please make sure to reopen the application?";
        confirm = "Confirm";
        author = "Author";
        evansLiu = "Evans Liu";
        about_info = " This software is a supporting software for institutional optical cameras. \n \n If you need GPU acceleration, please compile it from the source code! \n \n Please do not use this software for any unauthorized commercial purposes, including sales, promotion, etc! \n \n Follow the official account: actual 3D vision. Share more technical content with you.";
        wechat = " wechat";
        online = "Online";
        offline = "Offline"
        monocular_sl_camera = "Monocular S-L Camera";
        binocular_sl_camera = "Binocular S-L Camera"
        triple_sl_camera = "Triple S-L Camera";
        connect = "Connect";
        stripe_encoding = "Stripe Encoding"
        pixel_depth = "Pixel Depth"
        one_bit = "One Bit";
        eight_bit = "Eight Bit";
        camera = "Camera";
        stripe_type = "Stripe Type";
        sine_complementary_gray_code = "Sine Complementary Gray Code";
        sine_shift_gray_code = "Sine Shift Gray Code";
        encode = "Encode";
        save = "Save";
        defocus_encoding = "Defocus Encoding";
        opwm = "Optimal Pulse Width Modulation";
        error_diffusion_method = "Error Diffusion Method";
        img_width = "IMG Width";
        img_height = "IMG Height";
        clip_width = "Clip Width";
        clip_height = "Clip Height";
        cycles = "Cycles";
        shift_time = "Shift Time";
        stripe_img = "Stripe IMG";
        burn = "Burn";
        previous_text = "<Previous";
        next_text = "Next>";
        direction = "Direction";
        honrizon = "Honrizon";
        vertical = "Vertical";
        connect_sucess = "Connect Sucess!";
        connect_failed = "Connect Failed!";
        disconnect_sucess = "Disconnect Sucess!";
        disconnect_failed = "Disconnect Failed!";
        connect_ing = "Connecting...";
        burn_ing = "Burning Stripe ing...";
        burn_finished = "Burning Stripe finished!";
        disconnect = "Disconnect";
        disconnect_ing = "Disconnect ing...";
        save_file = "Save File";
        please_choose_save_folder = "Please Selcet folder to save file...";
        disable = "Disable";
        binary = "Binary";
        one_bit_disable_defocus_warning = "1 bit stripe encoding must be encoded by choosing the defocus method.";
        calibration_type_select = "Calibration Type Select:";
        camera_offline_calibration = "Offline Camera Calibration";
        camera_offline_calibration_tip = "Offline camera calibration: Extract feature points and perform monocular and bi-target calibration through the offline acquisition of calibration plate images.";
        camera_online_calibration = "Online Camera Calibration";
        camera_online_calibration_tip = "Online camera calibration collects the image of the calibration plate online through the camera, extracts the feature points and performs monocular and bi-target calibration.";
        projector_online_calibration = "Online Projector Calibration";
        projector_online_calibration_tip = "Online projector calibration collects the sinusoidal Gray code pattern projected by the projector and dephasing, so that the camera helps the projector to collect the calibration plate picture by the auxiliary camera reverse projection method. The program generates an eight-parameter calibration file as well as projector internal parameters.";
        img_browser = "IMG Browser";
        please_select_left_folder = "Please select left folder...";
        please_select_right_folder = "Please select right folder...";
        please_select_color_folder = "Please select color Folder";
        select_left_folder = "Select Left Folder";
        select_right_folder = "Select Right Folder";
        select_color_folder = "Select Color Folder";
        left_camera = "Left Camera";
        right_camera = "Right Camera";
        color_camera = "Color Camera";
        remove = "Remove";
        target_type = "Target Type";
        row_feature_points_num = "Row Feature Points";
        col_feature_points_num = "Col Feature Points";
        features_distance = "Features Distance";
        inner_circle_inner_radius = "inner circle inner radius";
        inner_circle_externer_radius = "inner circle externer radius";
        externer_circle_inner_radius = "externer circle inner radius";
        externer_circle_externer_radius = "externer circle externer radius";
        chess_board = "Chess Board";
        blob = "Blob";
        concentric_circle = "Concentric Circle";
        single_calibration = "Single Calibration";
        stereo_calibration = "Stereo Calibration";
        triple_calibration = "Triple Calibration";
        calibration_ing = "Calibration-ing...";
        calibration_error = "Calibration error: ";
        save_sucess = "Save success!";
        error_point_distribution = "Cuurent IMG Error Point Distribution";
        img_error_distribution = "Reproject RMSE: ";
        result_display = "Calibration Result Display";
        display_rectify_img = "Rectified IMG";
        export_as_left_color_calibration = "Exported as a left-color camera calibration parameter";
        capture = "Capture";
        set_calibration_params = "Set Calibration Params";
        cam_img = "Camera IMG";
        phase_h = "Horizon Phase";
        phase_v = "Vertical Phase";
        projector_img = "Projector IMG";
        honrizon_pitch = "Honrizon Pitch";
        vertical_pitch = "Vertical Pitch";
        calibrate_intrincis = "Calibrate Intrinsics";
        calibrate_extrincis = "Calibrate Extrinsics";
        capture_sucess = "Capture successful!";
        capture_failed = "Capture fails! Check whether the feature points cannot be extracted successfully!";
        projector_camera = "Projector";
        static_scan_mode = "Static Scan Mode";
        dynamic_scan_mode = "Dynamic Scan Mode";
        offline_scan_mode = "Offline Scan Mode";
        static_scan_mode_tip = "The static scan mode enables data processing with greater accuracy and ease of operation.";
        dynamic_scan_mode_tip = "The dynamic scanning mode can run at the highest 3D reconstruction rate of the system, and has the ability to reconstruct dynamic objects in 3D.";
        offline_scan_mode_tip = "The offline scan mode makes it easier to work with offline data, and it also makes it easier for developers to develop their own algorithms.";
        scan_mode_selcet = "Scan Mode Select: ";
        start_scan = "Start Scan";
        light_strength = "Light Strength";
        exposure_time = "Exposure Time";
        filter_threshod = "Filter Threshod";
        pattern_type = "Pattern Type";
        enable_gpu = "GPU Accelerate";
        texture = "Texture";
        sinus_comple_gray_code = "Sinusoidal Complementary Gray Code";
        three_frequency_heterodyne = "Three Frequency Heterodyne";
        reserved = "Reserved";
        please_select = "Please select end point through right button click to finish draw bounding box!";
        select_finished = "Select bounding box finish!";
        please_select_start_point = "Please left-click on the screen to start drawing the bounding box!"
        please_select_point_to_see_info = "Please click on the dot for specific information!";
        multi_view_stereo_geometry = "Multi-View Stereo Geometry";
        find_features_error = "The image could not find the feature point, please check the parameters and the image!Path is: %1";
        pure_scan_mode = "Pure Scan";
        turret_scan_mode = "Turret Scan";
        configure_static_scan_algorithm = "Configure Static Scan Algorithm";
        pure_scan_desc = "Pure scan mode, no auxiliary devices or auxiliary algorithms are used.";
        turret_scan_desc = "Turntable scanning mode, using turntable auxiliary equipment to rotate and splice any object multiple times.";
        device_function_select = "Function Selcet";
        pre_exposure_time = "Pre Exposure Time";
        aft_exposure_time = "Aft Exposure Time";
        project_once = "Project Once";
        project_continue = "Project Continues";
        project_step = "Step Project";
        project_ten_line = "Ten Line";
        stop_project = "Stop Project";
        pause_project = "Pause Project";
        clear_pre_stripes = "Clear Pre Stripes";
        keep_add_stripes = "Keep Add Stripes";
        operation_dialog = "Operation Dialog";
        execute = "execute";
        nodes = "Nodes";
        inoutput = "Input/Output";
        filters = "Filters";
        registration = "Registration";
        sample_consensus = "Sample Consensus";
        segmentation = "Segmentation";
        surface = "Surface";
        features = "Features";
        cloudInputNode = "Cloud Input Node";
        cloudOutputNode = "Cloud Output Node";
        passThroughFilterNode = "Pass Through Filter Node";
        cloudInputMode = "Cloud Input Mode";
        selectCloudFile = "Select Cloud File";
        fromFile = "From File";
        fromCamera = "From Camera";
        enableX = "Enable X";
        minX = "Min X";
        maxX = "Max X";
        enableY = "Enable Y";
        minY = "Min Y";
        maxY = "Max Y";
        enableZ = "Enable Z";
        minZ = "Min Z";
        maxZ = "Max Z";
        staticRemovel = "Statistical Removal Node";
        meanK = "Number of closest neighbors";
        stdThresh = "Standard deviation coefficient";
        group = "Group";
        greedyProjectionTriangulation = "Greedy Projection Triangulation Node";
        kSearch = "K";
        multiplier = "Multiplier";
        maxNearestNumber = "Max Nearest Number";
        searchRadius = "Search Radius";
        minimumAngle = "Min Triangle Angle";
        maximumAngle = "Max Triangle Angle";
        maximumSurfaceAngle = "Max Surface Angle";
        normalConsistency = "Normal Consistency";
        meshOutputNode = "Mesh Output Node";
        poisson = "Poisson Node";
        minDepth = "Min Depth";
        maxDepth = "Max Depth";
        scale = "Scale";
        solverDivide = "Solver Divide";
        isoDivide = "Iso Divide";
        minSamplePoints = "Min Sample Points";
        degree = "Degree";
        confidence = "Confidence";
        manifold = "Manifold";
        img_display = "IMG Display";
        read_params = "Read Params";
        export_epiline = "Export Epilines";
        model_type = "Model Type";
        method_type = "Method Type";
        distance_threshold = "Distance Threshold";
        sac_RANSAC = "SAC_RANSAC";
        sac_LMEDS = "SAC_LMEDS";
        sac_MSAC = "SAC_MSAC";
        sac_RRANSAC = "SAC_RRANSAC";
        sac_RMSAC = "SAC_RMSAC";
        sac_MLESAC = "SAC_MLESAC";
        sac_PROSAC = "SAC_PROSAC";
        sacMODEL_PLANE = "SACMODEL_PLANE";
        sacMODEL_LINE = "SACMODEL_LINE";
        sacMODEL_CIRCLE2D = "SACMODEL_CIRCLE2D";
        sacMODEL_CIRCLE3D = "SACMODEL_CIRCLE3D";
        sacMODEL_SPHERE = "SACMODEL_SPHERE";
        sacMODEL_CYLINDER = "SACMODEL_CYLINDER";
        sacMODEL_CONE = "SACMODEL_CONE";
        sacMODEL_TORUS = "SACMODEL_TORUS";
        sacMODEL_PARALLEL_LINE = "SACMODEL_PARALLEL_LINE";
        sacMODEL_PERPENDICULAR_PLANE = "SACMODEL_PERPENDICULAR_PLANE";
        sacMODEL_PARALLEL_LINES = "SACMODEL_PARALLEL_LINES";
        sacMODEL_NORMAL_PLANE = "SACMODEL_NORMAL_PLANE";
        sacMODEL_NORMAL_SPHERE = "SACMODEL_NORMAL_SPHERE";
        sacMODEL_REGISTRATION = "SACMODEL_REGISTRATION";
        sacMODEL_REGISTRATION_2D = "SACMODEL_REGISTRATION_2D";
        sacMODEL_PARALLEL_PLANE = "SACMODEL_PARALLEL_PLANE";
        sacMODEL_NORMAL_PARALLEL_LINES = "SACMODEL_NORMAL_PARALLEL_LINES";
        sacMODEL_STICK = "SACMODEL_STICK";
        sacMODEL_ELLIPSE3D = "SACMODEL_ELLIPSE3D";
        sac_segment = "SAC Segment";
        split_output_node = "Split Output Node";
        split_port_id = "Split Port ID";
        intersection_line_node = "Intersection Line Node";
        three_line_intersection_node = "Three Line Intersection Node";
        actor_output_node = "Actor Output Node";
        width = "Width";
        height = "Height";
        length = "Length";
        sequenceInverse = "Feature Point Sequence Inverse";
        horizonVerticalInv = "Horizon/Vertical Feature Point Sequence Inverse";
        useCurrentFeaturePoints = "Use Current Feature Points";
        interzone_sinus_four_grayscale = "Interzone Sinus Four Grayscale";
    }

    property string __locale
    property var __localeList: ["Zh", "En"]

    on__LocaleChanged: {
        if(__locale === "Zh") {
            zh();
        }
        else {
            en();
        }
    }

    Component.onCompleted: {
        __locale = "En";
    }
}
