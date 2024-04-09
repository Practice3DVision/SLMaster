#include <gtest/gtest.h>

#include <projectorFactory.h>

const std::string testProjector3010 = "DLP3010";
const std::string testData3010 = "../../data/4_3010";

using namespace slmaster::device;

TEST(Projector, init) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    ASSERT_NE(projectorDlpcApi, nullptr);
}

TEST(Project, connect) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Project, isConnect) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->isConnect();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, disconnect) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    ASSERT_EQ(projectorDlpcApi->disConnect(), true);
    projectorDlpcApi->disConnect();
}

TEST(Project, onceProject) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(false);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, continueProject) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, pause) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->pause();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, resume) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->pause();
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    isSucess = projectorDlpcApi->resume();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}

TEST(Project, stop) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->stop();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Project, step) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->project(true);
    isSucess = projectorDlpcApi->step();
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Project, getLEDCurrent) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    double red, green, blue;
    isSucess = projectorDlpcApi->getLEDCurrent(red, green, blue);
    printf("projector's current light strength: red %d, green %d, blue %d", red, green, blue);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Project, setLEDCurrent) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();
    isSucess = projectorDlpcApi->setLEDCurrent(0.95, 0.95, 0.95);
    double red, green, blue;
    isSucess = projectorDlpcApi->getLEDCurrent(red, green, blue);
    printf("after set light stength, projector's current light strength: red %d, green %d, blue %d", red, green, blue);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->disConnect();
}

TEST(Projector, populatePatternTableData) {
    auto projectorFactory = ProjectorFactory();
    auto projectorDlpcApi = projectorFactory.getProjector(testProjector3010);
    auto info = projectorDlpcApi->getInfo();
    bool isSucess = projectorDlpcApi->connect();

    std::vector<PatternOrderSet> patternSets(2);
    patternSets[0].exposureTime_ = 20000;
    patternSets[0].preExposureTime_ = 3000;
    patternSets[0].postExposureTime_ = 3000;
    patternSets[0].illumination_ = Blue;
    patternSets[0].invertPatterns_ = false;
    patternSets[0].isVertical_ = true;
    patternSets[0].isOneBit_ = false;
    patternSets[0].patternArrayCounts_ = 1280;
    patternSets[1].exposureTime_ = 20000;
    patternSets[1].preExposureTime_ = 3000;
    patternSets[1].postExposureTime_ = 3000;
    patternSets[1].illumination_ = Blue;
    patternSets[1].invertPatterns_ = false;
    patternSets[1].isVertical_ = true;
    patternSets[1].isOneBit_ = false;
    patternSets[1].patternArrayCounts_ = 1280;

    std::vector<cv::String> imgsPaths;
    cv::glob(testData3010, imgsPaths);
    std::vector<cv::Mat> imgFirstSet;
    std::vector<cv::Mat> imgSecondSet;
    for (size_t i = 0; i < imgsPaths.size(); ++i) {
        imgFirstSet.push_back(cv::imread(testData3010 + "/" + std::to_string(i) + ".bmp", 0));
        imgSecondSet.push_back(cv::imread(testData3010 + "/" + std::to_string(i) + ".bmp", 0));
    }
    patternSets[0].imgs_ = imgFirstSet;
    patternSets[1].imgs_ = imgSecondSet;

    isSucess = projectorDlpcApi->populatePatternTableData(patternSets);
    ASSERT_EQ(isSucess, true);
    projectorDlpcApi->project(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    projectorDlpcApi->stop();
    projectorDlpcApi->disConnect();
}
