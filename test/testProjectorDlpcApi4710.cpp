#include <gtest/gtest.h>

#include <projectorFactory.h>

const std::string testProjector4710 = "DLP4710";
const std::string testData4710 = "../../data/4_4710";

using namespace slmaster::device;

TEST(Projector, init) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    ASSERT_NE(projectorDlpcApi, nullptr);
}

TEST(Projector, getInfo) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    ASSERT_EQ(info.isFind_, true);
}

TEST(Project, connect) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->disConnect();
}

TEST(Project, isConnect) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->isConnect();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Project, disconnect) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    ASSERT_EQ(projectorDlpcApi->disConnect(), true);
}

TEST(Project, onceProject) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, continueProject) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    isSucess = projectorDlpcApi->disConnect();
    ASSERT_EQ(isSucess, true);
}

TEST(Project, pause) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    isSucess = projectorDlpcApi->pause();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    isSucess = projectorDlpcApi->disConnect();
    ASSERT_EQ(isSucess, true);
}

TEST(Project, resume) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    isSucess = projectorDlpcApi->resume();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    isSucess = projectorDlpcApi->disConnect();
    ASSERT_EQ(isSucess, true);
}

TEST(Project, stop) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    isSucess = projectorDlpcApi->stop();
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->disConnect();
    ASSERT_EQ(isSucess, true);
}

TEST(Projector, populatePatternTableData) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();

    ASSERT_EQ(isSucess, true);

    std::vector<PatternOrderSet> patternSets(2);
    patternSets[0].exposureTime_ = 4000;
    patternSets[0].preExposureTime_ = 3000;
    patternSets[0].postExposureTime_ = 3000;
    patternSets[0].illumination_ = Blue;
    patternSets[0].invertPatterns_ = false;
    patternSets[0].isVertical_ = true;
    patternSets[0].isOneBit_ = false;
    patternSets[0].patternArrayCounts_ = 1920;
    patternSets[1].exposureTime_ = 4000;
    patternSets[1].preExposureTime_ = 3000;
    patternSets[1].postExposureTime_ = 3000;
    patternSets[1].illumination_ = Blue;
    patternSets[1].invertPatterns_ = false;
    patternSets[1].isVertical_ = true;
    patternSets[1].isOneBit_ = false;
    patternSets[1].patternArrayCounts_ = 1920;

    std::vector<cv::String> imgsPaths;
    cv::glob(testData4710, imgsPaths);
    std::vector<cv::Mat> imgFirstSet;
    std::vector<cv::Mat> imgSecondSet;
    for (size_t i = 0; i < imgsPaths.size() / 2; ++i) {
        imgFirstSet.push_back(cv::imread(testData4710 + "/IMG_" + std::to_string(i) + ".bmp", 0));
        imgSecondSet.push_back(cv::imread(testData4710 + "/IMG_" + std::to_string(i + 5) + ".bmp", 0));
    }
    patternSets[0].imgs_ = imgFirstSet;
    patternSets[1].imgs_ = imgSecondSet;

    ASSERT_EQ(patternSets[0].imgs_.empty(), false);
    ASSERT_EQ(patternSets[1].imgs_.empty(), false);

    isSucess = projectorDlpcApi->populatePatternTableData(patternSets);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->project(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, step) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    isSucess = projectorDlpcApi->step();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    isSucess = projectorDlpcApi->step();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    isSucess = projectorDlpcApi->step();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    isSucess = projectorDlpcApi->step();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    isSucess = projectorDlpcApi->stop();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->disConnect();
    ASSERT_EQ(isSucess, true);
}

TEST(Project, getSetLEDCurrent) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector4710);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    double red, green, blue;
    isSucess = projectorDlpcApi->getLEDCurrent(red, green, blue);
    printf("projector's current light strength: red %f, green %f, blue %f \n", red, green, blue);
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->setLEDCurrent(0.95, 0.95, 0.95);
    ASSERT_EQ(isSucess, true);
    isSucess = projectorDlpcApi->getLEDCurrent(red, green, blue);
    printf("after set light stength, projector's current light strength: red %f, green %f, blue %f \n", red, green, blue);
    isSucess = projectorDlpcApi->disConnect();
    ASSERT_EQ(isSucess, true);
}
