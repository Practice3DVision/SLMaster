#include "edgesSubPix.h"

#include <cmath>
#include <execution>
#include <numeric>

using namespace cv;
using namespace std;
using namespace slmaster::calibration;

const double scale = 128.0; // sum of half Canny filter is 128

static void getCannyKernel(OutputArray _d, double alpha) {
    int r = cvRound(alpha * 3);
    int ksize = 2 * r + 1;

    _d.create(ksize, 1, CV_16S, -1, true);

    Mat k = _d.getMat();

    vector<float> kerF(ksize, 0.0f);
    kerF[r] = 0.0f;
    double a2 = alpha * alpha;
    float sum = 0.0f;
    for (int x = 1; x <= r; ++x) {
        float v = (float)(-x * std::exp(-x * x / (2 * a2)));
        sum += v;
        kerF[r + x] = v;
        kerF[r - x] = -v;
    }
    float scale = 128 / sum;
    for (int i = 0; i < ksize; ++i) {
        kerF[i] *= scale;
    }
    Mat temp(ksize, 1, CV_32F, &kerF[0]);
    temp.convertTo(k, CV_16S);
}

// non-maximum supression and hysteresis
static void postCannyFilter(const Mat &src, Mat &dx, Mat &dy, int low, int high,
                            Mat &dst) {
    ptrdiff_t mapstep = src.cols + 2;
    AutoBuffer<uchar> buffer((src.cols + 2) * (src.rows + 2) +
                             mapstep * 3 * sizeof(int));

    // L2Gradient comparison with square
    high = high * high;
    low = low * low;

    int *mag_buf[3];
    mag_buf[0] = (int *)(uchar *)buffer;
    mag_buf[1] = mag_buf[0] + mapstep;
    mag_buf[2] = mag_buf[1] + mapstep;
    memset(mag_buf[0], 0, mapstep * sizeof(int));

    uchar *map = (uchar *)(mag_buf[2] + mapstep);
    memset(map, 1, mapstep);
    memset(map + mapstep * (src.rows + 1), 1, mapstep);

    int maxsize = std::max(1 << 10, src.cols * src.rows / 10);
    std::vector<uchar *> stack(maxsize);
    uchar **stack_top = &stack[0];
    uchar **stack_bottom = &stack[0];

    /* sector numbers
    (Top-Left Origin)

    1   2   3
    *  *  *
    * * *
    0*******0
    * * *
    *  *  *
    3   2   1
    */

#define CANNY_PUSH(d) *(d) = uchar(2), *stack_top++ = (d)
#define CANNY_POP(d) (d) = *--stack_top

#if CV_SSE2
    bool haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
#endif

    // calculate magnitude and angle of gradient, perform non-maxima
    // suppression. fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for (int i = 0; i <= src.rows; i++) {
        int *_norm = mag_buf[(i > 0) + 1] + 1;
        if (i < src.rows) {
            short *_dx = dx.ptr<short>(i);
            short *_dy = dy.ptr<short>(i);

            int j = 0, width = src.cols;
#if CV_SSE2
            if (haveSSE2) {
                for (; j <= width - 8; j += 8) {
                    __m128i v_dx = _mm_loadu_si128((const __m128i *)(_dx + j));
                    __m128i v_dy = _mm_loadu_si128((const __m128i *)(_dy + j));

                    __m128i v_dx_ml = _mm_mullo_epi16(v_dx, v_dx),
                            v_dx_mh = _mm_mulhi_epi16(v_dx, v_dx);
                    __m128i v_dy_ml = _mm_mullo_epi16(v_dy, v_dy),
                            v_dy_mh = _mm_mulhi_epi16(v_dy, v_dy);

                    __m128i v_norm =
                        _mm_add_epi32(_mm_unpacklo_epi16(v_dx_ml, v_dx_mh),
                                      _mm_unpacklo_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j), v_norm);

                    v_norm =
                        _mm_add_epi32(_mm_unpackhi_epi16(v_dx_ml, v_dx_mh),
                                      _mm_unpackhi_epi16(v_dy_ml, v_dy_mh));
                    _mm_storeu_si128((__m128i *)(_norm + j + 4), v_norm);
                }
            }
#elif CV_NEON
            for (; j <= width - 8; j += 8) {
                int16x8_t v_dx = vld1q_s16(_dx + j), v_dy = vld1q_s16(_dy + j);
                int16x4_t v_dxp = vget_low_s16(v_dx),
                          v_dyp = vget_low_s16(v_dy);
                int32x4_t v_dst =
                    vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j, v_dst);

                v_dxp = vget_high_s16(v_dx), v_dyp = vget_high_s16(v_dy);
                v_dst = vmlal_s16(vmull_s16(v_dxp, v_dxp), v_dyp, v_dyp);
                vst1q_s32(_norm + j + 4, v_dst);
            }
#endif
            for (; j < width; ++j)
                _norm[j] = int(_dx[j]) * _dx[j] + int(_dy[j]) * _dy[j];

            _norm[-1] = _norm[src.cols] = 0;
        } else
            memset(_norm - 1, 0, /* cn* */ mapstep * sizeof(int));

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if (i == 0)
            continue;

        uchar *_map = map + mapstep * i + 1;
        _map[-1] = _map[src.cols] = 1;

        int *_mag = mag_buf[1] + 1; // take the central row
        ptrdiff_t magstep1 = mag_buf[2] - mag_buf[1];
        ptrdiff_t magstep2 = mag_buf[0] - mag_buf[1];

        const short *_x = dx.ptr<short>(i - 1);
        const short *_y = dy.ptr<short>(i - 1);

        if ((stack_top - stack_bottom) + src.cols > maxsize) {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = std::max(maxsize * 3 / 2, sz + src.cols);
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        int prev_flag = 0;
        for (int j = 0; j < src.cols; j++) {
#define CANNY_SHIFT 15
            const int TG22 =
                (int)(0.4142135623730950488016887242097 * (1 << CANNY_SHIFT) +
                      0.5);

            int m = _mag[j];

            if (m > low) {
                int xs = _x[j];
                int ys = _y[j];
                int x = std::abs(xs);
                int y = std::abs(ys) << CANNY_SHIFT;

                int tg22x = x * TG22;

                if (y < tg22x) {
                    if (m > _mag[j - 1] && m >= _mag[j + 1])
                        goto __ocv_canny_push;
                } else {
                    int tg67x = tg22x + (x << (CANNY_SHIFT + 1));
                    if (y > tg67x) {
                        if (m > _mag[j + magstep2] && m >= _mag[j + magstep1])
                            goto __ocv_canny_push;
                    } else {
                        int s = (xs ^ ys) < 0 ? -1 : 1;
                        if (m > _mag[j + magstep2 - s] &&
                            m > _mag[j + magstep1 + s])
                            goto __ocv_canny_push;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = uchar(1);
            continue;
        __ocv_canny_push:
            if (!prev_flag && m > high && _map[j - mapstep] != 2) {
                CANNY_PUSH(_map + j);
                prev_flag = 1;
            } else
                _map[j] = 0;
        }

        // scroll the ring buffer
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
    }

    // now track the edges (hysteresis thresholding)
    while (stack_top > stack_bottom) {
        uchar *m;
        if ((stack_top - stack_bottom) + 8 > maxsize) {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = maxsize * 3 / 2;
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);

        if (!m[-1])
            CANNY_PUSH(m - 1);
        if (!m[1])
            CANNY_PUSH(m + 1);
        if (!m[-mapstep - 1])
            CANNY_PUSH(m - mapstep - 1);
        if (!m[-mapstep])
            CANNY_PUSH(m - mapstep);
        if (!m[-mapstep + 1])
            CANNY_PUSH(m - mapstep + 1);
        if (!m[mapstep - 1])
            CANNY_PUSH(m + mapstep - 1);
        if (!m[mapstep])
            CANNY_PUSH(m + mapstep);
        if (!m[mapstep + 1])
            CANNY_PUSH(m + mapstep + 1);
    }

    // the final pass, form the final image
    const uchar *pmap = map + mapstep + 1;
    uchar *pdst = dst.ptr();
    for (int i = 0; i < src.rows; i++, pmap += mapstep, pdst += dst.step) {
        for (int j = 0; j < src.cols; j++)
            pdst[j] = (uchar) - (pmap[j] >> 1);
    }
}

static inline double getAmplitude(Mat &dx, Mat &dy, int i, int j) {
    Point2d mag(dx.at<short>(i, j), dy.at<short>(i, j));
    return norm(mag);
}

static inline void getMagNeighbourhood(Mat &dx, Mat &dy, Point &p, int w, int h,
                                       vector<double> &mag) {
    int top = p.y - 1 >= 0 ? p.y - 1 : p.y;
    int down = p.y + 1 < h ? p.y + 1 : p.y;
    int left = p.x - 1 >= 0 ? p.x - 1 : p.x;
    int right = p.x + 1 < w ? p.x + 1 : p.x;

    mag[0] = getAmplitude(dx, dy, top, left);
    mag[1] = getAmplitude(dx, dy, top, p.x);
    mag[2] = getAmplitude(dx, dy, top, right);
    mag[3] = getAmplitude(dx, dy, p.y, left);
    mag[4] = getAmplitude(dx, dy, p.y, p.x);
    mag[5] = getAmplitude(dx, dy, p.y, right);
    mag[6] = getAmplitude(dx, dy, down, left);
    mag[7] = getAmplitude(dx, dy, down, p.x);
    mag[8] = getAmplitude(dx, dy, down, right);
}

static inline void get2ndFacetModelIn3x3(vector<double> &mag,
                                         vector<double> &a) {
    a[0] = (-mag[0] + 2.0 * mag[1] - mag[2] + 2.0 * mag[3] + 5.0 * mag[4] +
            2.0 * mag[5] - mag[6] + 2.0 * mag[7] - mag[8]) /
           9.0;
    a[1] = (-mag[0] + mag[2] - mag[3] + mag[5] - mag[6] + mag[8]) / 6.0;
    a[2] = (mag[6] + mag[7] + mag[8] - mag[0] - mag[1] - mag[2]) / 6.0;
    a[3] = (mag[0] - 2.0 * mag[1] + mag[2] + mag[3] - 2.0 * mag[4] + mag[5] +
            mag[6] - 2.0 * mag[7] + mag[8]) /
           6.0;
    a[4] = (-mag[0] + mag[2] + mag[6] - mag[8]) / 4.0;
    a[5] = (mag[0] + mag[1] + mag[2] - 2.0 * (mag[3] + mag[4] + mag[5]) +
            mag[6] + mag[7] + mag[8]) /
           6.0;
}
/*
   Compute the eigenvalues and eigenvectors of the Hessian matrix given by
   dfdrr, dfdrc, and dfdcc, and sort them in descending order according to
   their absolute values.
*/
static inline void eigenvals(vector<double> &a, double eigval[2],
                             double eigvec[2][2]) {
    // derivatives
    // fx = a[1], fy = a[2]
    // fxy = a[4]
    // fxx = 2 * a[3]
    // fyy = 2 * a[5]
    double dfdrc = a[4];
    double dfdcc = a[3] * 2.0;
    double dfdrr = a[5] * 2.0;
    double theta, t, c, s, e1, e2, n1, n2; /* , phi; */

    /* Compute the eigenvalues and eigenvectors of the Hessian matrix. */
    if (dfdrc != 0.0) {
        theta = 0.5 * (dfdcc - dfdrr) / dfdrc;
        t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
        if (theta < 0.0)
            t = -t;
        c = 1.0 / sqrt(t * t + 1.0);
        s = t * c;
        e1 = dfdrr - t * dfdrc;
        e2 = dfdcc + t * dfdrc;
    } else {
        c = 1.0;
        s = 0.0;
        e1 = dfdrr;
        e2 = dfdcc;
    }
    n1 = c;
    n2 = -s;

    /* If the absolute value of an eigenvalue is larger than the other, put that
    eigenvalue into first position.  If both are of equal absolute value, put
    the negative one first. */
    if (fabs(e1) > fabs(e2)) {
        eigval[0] = e1;
        eigval[1] = e2;
        eigvec[0][0] = n1;
        eigvec[0][1] = n2;
        eigvec[1][0] = -n2;
        eigvec[1][1] = n1;
    } else if (fabs(e1) < fabs(e2)) {
        eigval[0] = e2;
        eigval[1] = e1;
        eigvec[0][0] = -n2;
        eigvec[0][1] = n1;
        eigvec[1][0] = n1;
        eigvec[1][1] = n2;
    } else {
        if (e1 < e2) {
            eigval[0] = e1;
            eigval[1] = e2;
            eigvec[0][0] = n1;
            eigvec[0][1] = n2;
            eigvec[1][0] = -n2;
            eigvec[1][1] = n1;
        } else {
            eigval[0] = e2;
            eigval[1] = e1;
            eigvec[0][0] = -n2;
            eigvec[0][1] = n1;
            eigvec[1][0] = n1;
            eigvec[1][1] = n2;
        }
    }
}

static inline double vector2angle(double x, double y) {
    double a = std::atan2(y, x);
    return a >= 0.0 ? a : a + CV_2PI;
}

void extractSubPixPointsSteger(Mat &dx, Mat &dy,
                               vector<vector<Point>> &contoursInPixel,
                               vector<Contour> &contours) {
    int w = dx.cols;
    int h = dx.rows;
    contours.resize(contoursInPixel.size());
    for (size_t i = 0; i < contoursInPixel.size(); ++i) {
        vector<Point> &icontour = contoursInPixel[i];
        Contour &contour = contours[i];
        contour.points.resize(icontour.size());
        contour.response.resize(icontour.size());
        contour.direction.resize(icontour.size());
#if defined(_OPENMP) && defined(NDEBUG)
#pragma omp parallel for
#endif
        for (int j = 0; j < (int)icontour.size(); ++j) {
            vector<double> magNeighbour(9);
            getMagNeighbourhood(dx, dy, icontour[j], w, h, magNeighbour);
            vector<double> a(9);
            get2ndFacetModelIn3x3(magNeighbour, a);

            // Hessian eigen vector
            double eigvec[2][2], eigval[2];
            eigenvals(a, eigval, eigvec);
            double t = 0.0;
            double ny = eigvec[0][0];
            double nx = eigvec[0][1];
            if (eigval[0] < 0.0) {
                double rx = a[1], ry = a[2], rxy = a[4], rxx = a[3] * 2.0,
                       ryy = a[5] * 2.0;
                t = -(rx * nx + ry * ny) /
                    (rxx * nx * nx + 2.0 * rxy * nx * ny + ryy * ny * ny);
            }
            double px = nx * t;
            double py = ny * t;
            float x = (float)icontour[j].x;
            float y = (float)icontour[j].y;
            if (fabs(px) <= 0.5 && fabs(py) <= 0.5) {
                x += (float)px;
                y += (float)py;
            }
            contour.points[j] = Point2f(x, y);
            contour.response[j] = (float)(a[0] / scale);
            contour.direction[j] = (float)vector2angle(ny, nx);
        }
    }
}

void extractSubPixPointsDevernay(Mat &dx, Mat &dy,
                                 vector<vector<Point>> &contoursInPixel,
                                 vector<Contour> &contours) {
    int w = dx.cols;
    int h = dx.rows;
    contours.resize(contoursInPixel.size());
    for (size_t i = 0; i < contoursInPixel.size(); ++i) {
        vector<Point> &icontour = contoursInPixel[i];
        Contour &contour = contours[i];
        contour.points.resize(icontour.size());
        contour.response.resize(icontour.size());
        contour.direction.resize(icontour.size());
#if defined(_OPENMP) && defined(NDEBUG)
#pragma omp parallel for
#endif
        for (int j = 0; j < (int)icontour.size(); ++j) {
            int x = icontour[j].x;
            int y = icontour[j].y;

            if (x + 1 > w - 1 || x - 1 < 0 || y - 1 < 0 || y + 1 > h - 1) {
                contour.points[j] = icontour[j];
                continue;
            }

            int Dx = 0; /* interpolation is along Dx,Dy		*/
            int Dy = 0; /* which will be selected below		*/
            double mod = std::sqrt(dx.ptr<short>(y)[x] * dx.ptr<short>(y)[x] +
                                   dy.ptr<short>(y)[x] *
                                       dy.ptr<short>(y)[x]); /* modG at pixel */
            double L = std::sqrt(
                dx.ptr<short>(y)[x - 1] * dx.ptr<short>(y)[x - 1] +
                dy.ptr<short>(y)[x - 1] *
                    dy.ptr<short>(y)[x - 1]); /* modG at pixel on the left */
            double R = std::sqrt(
                dx.ptr<short>(y)[x + 1] * dx.ptr<short>(y)[x + 1] +
                dy.ptr<short>(y)[x + 1] *
                    dy.ptr<short>(y)[x + 1]); /* modG at pixel on the right
                                               */
            double U =
                std::sqrt(dx.ptr<short>(y + 1)[x] * dx.ptr<short>(y + 1)[x] +
                          dy.ptr<short>(y + 1)[x] *
                              dy.ptr<short>(y + 1)[x]); /* modG at pixel up */
            double D = std::sqrt(
                dx.ptr<short>(y - 1)[x] * dx.ptr<short>(y - 1)[x] +
                dy.ptr<short>(y - 1)[x] *
                    dy.ptr<short>(y - 1)[x]);      /* modG at pixel below */
            double gx = fabs(dx.ptr<short>(y)[x]); /* absolute value of Gx */
            double gy = fabs(dy.ptr<short>(y)[x]); /* absolute value of Gy */
            /* when local horizontal maxima of the gradient modulus and the
            gradient direction is more horizontal (|Gx| >= |Gy|),=> a
            "horizontal" (H) edge found else, if local vertical maxima of the
            gradient modulus and the gradient direction is more vertical (|Gx|
            <= |Gy|),=> a "vertical" (V) edge found */

            /* it can happen that two neighbor pixels have equal value and are
            both	maxima, for example when the edge is exactly between
            both pixels. in such cases, as an arbitrary convention, the edge is
            marked on the left one when an horizontal max or below when a
            vertical max. for	this the conditions are L < mod >= R and D < mod
            >= U,respectively. the comparisons are done using the function
            greater() instead of the operators > or >= so numbers differing only
            due to rounding errors are considered equal */
            if (mod > L && mod > R && gx >= gy)
                Dx = 1; /* H */
            if (mod > D && mod > U && gx <= gy)
                Dy = 1; /* V */

            /* Devernay sub-pixel correction

            the edge point position is selected as the one of the maximum of a
            quadratic interpolation of the magnitude of the gradient along a
            unidimensional direction. the pixel must be a local maximum. so we
            have the values:

            the x position of the maximum of the parabola passing through(-1,a),
            (0,b), and (1,c) is offset = (a - c) / 2(a - 2b + c),and because b
            >= a and b >= c, -0.5 <= offset <= 0.5	*/
            if (Dx > 0 || Dy > 0) {
                /* offset value is in [-0.5, 0.5] */
                int xSub = x - Dx;
                int ySub = y - Dy;
                int xAdd = x + Dx;
                int yAdd = y + Dy;
                double a = std::sqrt(
                    dx.ptr<short>(ySub)[xSub] * dx.ptr<short>(ySub)[xSub] +
                    dy.ptr<short>(ySub)[xSub] * dy.ptr<short>(ySub)[xSub]);
                double b = std::sqrt(dx.ptr<short>(y)[x] * dx.ptr<short>(y)[x] +
                                     dy.ptr<short>(y)[x] * dy.ptr<short>(y)[x]);
                double c = std::sqrt(
                    dx.ptr<short>(yAdd)[xAdd] * dx.ptr<short>(yAdd)[xAdd] +
                    dy.ptr<short>(yAdd)[xAdd] * dy.ptr<short>(yAdd)[xAdd]);
                double offset = 0.5 * (a - c) / (a - b - b + c);

                contour.points[j] =
                    cv::Point2f(x + offset * Dx, y + offset * Dy);
                contour.direction[j] = 0;
                contour.response[j] = 255;
            }
        }
    }

    vector<Contour> contourCopy(contours.begin(), contours.end());
    contours.clear();

    for (auto element : contourCopy) {
        Contour tour;
        tour.response = element.response;
        tour.direction = element.direction;

        for (auto pt : element.points) {
            if (pt.x < 0.01f && pt.y < 0.01f)
                continue;

            tour.points.emplace_back(pt);
        }

        contours.emplace_back(tour);
    }
}

//---------------------------------------------------------------------
//          INTERFACE FUNCTION
//---------------------------------------------------------------------
namespace slmaster {
namespace calibration {
void EdgesSubPix(Mat &gray, double alpha, int low, int high,
                 vector<Contour> &contours, OutputArray hierarchy, int mode) {
    Mat blur;
    GaussianBlur(gray, blur, Size(0, 0), alpha, alpha);

    Mat d;
    getCannyKernel(d, alpha);
    Mat one = Mat::ones(Size(1, 1), CV_16S);
    Mat dx, dy;
    sepFilter2D(blur, dx, CV_16S, d, one);
    sepFilter2D(blur, dy, CV_16S, one, d);

    // non-maximum supression & hysteresis threshold
    Mat edge = Mat::zeros(gray.size(), CV_8UC1);
    int lowThresh = cvRound(scale * low);
    int highThresh = cvRound(scale * high);
    postCannyFilter(gray, dx, dy, lowThresh, highThresh, edge);

    // contours in pixel precision
    vector<vector<Point>> contoursInPixel;
    findContours(edge, contoursInPixel, hierarchy, mode, CHAIN_APPROX_NONE);

    // subpixel position extraction with steger's method and facet model 2nd
    // polynominal in 3x3 neighbourhood
    // extractSubPixPointsSteger(dx, dy, contoursInPixel, contours);
    extractSubPixPointsDevernay(dx, dy, contoursInPixel, contours);
}

void EdgesSubPix(Mat &gray, double alpha, int low, int high,
                 vector<Contour> &contours) {
    vector<Vec4i> hierarchy;
    EdgesSubPix(gray, alpha, low, high, contours, hierarchy, RETR_LIST);
}
} // namespace calibration
} // namespace slmaster
