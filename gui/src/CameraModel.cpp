#include "cameraModel.h"

CameraModel::CameraModel(QObject *parent) : QAbstractListModel(parent) {
  roleNames_.insert(FileName, "FileName");
}

CameraModel::~CameraModel() {}

int CameraModel::rowCount(const QModelIndex &parent) const {
  Q_UNUSED(parent);
  return imgs_.count();
}

QVariant CameraModel::data(const QModelIndex &index, int role) const {
  if (!index.isValid()) return QVariant();

  switch (role) {
    case FileName: {
      return imgs_.value(index.row());
    }
    default:
      break;
  }

  return QVariant();
}

void CameraModel::emplace_back(const QString &imgInfo) {
  beginResetModel();

  imgs_.push_back(imgInfo);

  endResetModel();
}

int CameraModel::erase(const QString &imgInfo) {
  beginResetModel();

  int index = 0;
  auto iterator = imgs_.begin();
  for (; iterator != imgs_.end(); ++iterator) {
    if (imgInfo == *iterator) {
      imgs_.erase(iterator);
      break;
    }
    ++index;
  }

  endResetModel();

  if (iterator == imgs_.end() && index != imgs_.size()) return -1;

  return index;
}

void CameraModel::erase(int locIndex) {
  beginResetModel();

  imgs_.erase(imgs_.begin() + locIndex);

  endResetModel();
}

void CameraModel::recurseImg(const QString &folderUrl) {
  beginResetModel();

  auto folderPath = folderUrl.mid(8);

  curFolderPath_ = folderPath;

  std::vector<cv::String> imgPaths;
  std::vector<std::string> sortPaths;
  auto stdPath = curFolderPath_.toLocal8Bit().toStdString();
  cv::glob(stdPath + std::string("/*.bmp"), imgPaths);

  for (int i = 0; i < imgPaths.size(); ++i) {
    std::string::size_type iPos = imgPaths[i].find_last_of('\\') + 1;
    std::string filename = imgPaths[i].substr(iPos, imgPaths[i].length() - iPos);
    sortPaths.emplace_back(filename);
  }

  std::sort(sortPaths.begin(), sortPaths.end(), [](std::string a, std::string b) {
      std::string aName = a.substr(0, a.rfind("."));
      std::string bName = b.substr(0, b.rfind("."));

      return std::stoi(aName) < std::stoi(bName); }
    );

  imgs_.clear();
  for (auto &path : sortPaths) {
    size_t index = path.find_last_of("\\");
    if(index != -1) {
        path.replace(index, 1, "/");
    }
    imgs_.push_back(QString::fromStdString(path));
  }

  endResetModel();

  emit updateImgs();
}

QHash<int, QByteArray> CameraModel::roleNames() const { return roleNames_; }
