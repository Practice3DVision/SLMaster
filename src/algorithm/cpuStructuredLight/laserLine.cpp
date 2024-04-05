#include "lasterLine.h"

#include <Eigen/Eigen>

using namespace cv;
using namespace std;
using namespace Eigen;

namespace slmaster {
namespace algorithm {
void stegerExtract(const Mat &img, vector<Point2f> &outPoints,
                  InputArray &mask) {
    auto maskMat =
        mask.empty() ? Mat::ones(img.size(), CV_8UC1) : mask.getMat();

    Mutex mutex;

    parallel_for_(Range(0, img.rows), [&](const Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            if(i + 1 > img.rows - 1 || i + 2 > img.rows - 1 || i - 1 < 0 || i - 2 < 0) {
                continue;
            }

            auto ptrImg = img.ptr<uchar>(i);
            auto ptrImgLastRow = img.ptr<uchar>(i - 1);
            auto ptrImgNextRow = img.ptr<uchar>(i + 1);
            auto ptrMaskMat = maskMat.ptr<uchar>(i);

            for (int j = 0; j < img.cols; ++j) {
                if (ptrMaskMat[j] == 0 || j + 1 > img.cols - 1 || j + 2 > img.cols - 1 || j - 1 < 0 || j - 2 < 0) {
                    continue;
                }
                //前向差分
                auto fx = ptrImg[j + 1] - ptrImg[j];
                auto fy = ptrImgNextRow[j] - ptrImg[j];
                //中心差分
                auto fxx = ptrImg[j + 1] - 2 * ptrImg[j] + ptrImg[j - 1];
                auto fyy = ptrImgNextRow[j] - 2 * ptrImg[j] + ptrImgLastRow[j];
                auto fxy = ptrImgNextRow[j + 1] - ptrImg[j + 1] - ptrImgNextRow[j] + ptrImg[j];

                Matrix2f hessian;
                hessian << fxx, fxy, fxy, fyy;
                EigenSolver<Matrix2f> solver(hessian);
                auto eigenVal = solver.eigenvalues().real();
                auto eigenVec = solver.eigenvectors().real();

                int maxEigenIndex =
                    abs(eigenVal[0]) > abs(eigenVal[1]) ? 0 : 1;
                float nx = eigenVec.col(maxEigenIndex)(0);
                float ny = eigenVec.col(maxEigenIndex)(1);

                auto t = -(nx * fx + ny * fy) /
                         (nx * nx * fxx + 2.f * nx * ny * fxy + ny * ny * fyy);
                auto tnx = t * nx;
                auto tny = t * ny;

                if (tnx >= -0.5f && tnx <= 0.5f && tny >= -0.5f &&
                    tny <= 0.5f) {
                    lock_guard<Mutex> guard(mutex);
                    outPoints.emplace_back(Point2f(j + tnx, i + tny));
                }
            }
        }
    });
}
} // namespace algorithm
} // namespace slmaster