#include <iostream>
#include <dirent.h>
#include <string>
#include <opencv2/opencv.hpp>  // 包含 OpenCV 头文件

class train_cls {
public:
	train_cls(char * folder) {
		this->folder = folder;	
	}
	int resize() {
		DIR* dir = opendir(this->folder);
		if (dir!= nullptr) {
			struct dirent * entry;
			while ((entry = readdir(dir))!= nullptr) {
				cv::Mat img = cv::imread(entry->d_name, cv::IMREAD_COLOR);
 				if (img.empty()) {
					std::cerr << "无法读取图片。" << std::endl;
				} else {
					std::cout << "read picture success" << std::endl;
				}
    			}
				
		}
		closedir(dir);
		return 0;
	} 	
	int train();
	int result();
private:
	char * folder;
};

int main(int argc, char **argv)
{
	int result_class;
	train_cls entity(argv[1]);
	//entity.resize(folder)
	//enttiy.train();
	//result_class = enttiy.result();
	return 0;
}
