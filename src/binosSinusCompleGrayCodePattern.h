#ifndef __BINOSSINUSCOMPLEGRAYCODEPATTERN_H_
#define __BINOSSINUSCOMPLEGRAYCODEPATTERN_H_

#include "common.h"
#include "pattern.h"

namespace slmaster {
    //TODO@LiuYunhuang:使用修饰器模式会更好，避免未来方法增多导致的子类爆炸，需要在OpenCV中重新更改接口
    class SLMASTER_API BinoSinusCompleGrayCodePattern : public Pattern {
      public:
        BinoSinusCompleGrayCodePattern();
        virtual bool generate(IN std::vector<cv::Mat>& imgs) const override final;
        virtual bool decode(IN const std::vector< std::vector<cv::Mat> >& patternImages, OUT cv::Mat& disparityMap, IN const bool isGpu) const override final;
      private:
    };
}

#endif // __BINOSSINUSCOMPLEGRAYCODEPATTERN_H_
