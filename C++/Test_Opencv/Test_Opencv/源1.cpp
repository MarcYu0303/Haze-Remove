#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include<opencv2/core/core.hpp>
#include<iostream>
#include <vector>

using namespace cv;

#define CV_SORT_EVERY_ROW    0
#define CV_SORT_EVERY_COLUMN 1
#define CV_SORT_DESCENDING   16

void makeDepth32f(Mat& source, Mat& output)
{
	if ((source.depth() != CV_32F) > FLT_EPSILON)
		source.convertTo(output, CV_32F);
	else
		output = source;
}


void guidedFilter(Mat& source, Mat& guided_image, Mat& output, int radius, float epsilon)
{
	//CV_Assert(radius >= 2 && epsilon > 0);
	CV_Assert(source.data != NULL && source.channels() == 1);//�ɸı�����ͼ�����ͣ�ͨ����
	CV_Assert(guided_image.channels() == 1);                 //����ͼһ��Ϊ��ͨ��
	CV_Assert(source.rows == guided_image.rows && source.cols == guided_image.cols);

	Mat guided;
	if (guided_image.data == source.data)
		guided_image.copyTo(guided);
	else
		guided = guided_image;

	Mat source_32f, guided_32f;
	makeDepth32f(source, source_32f);//��������չΪ32λ�����ͣ��Ա��Ժ����˷�
	makeDepth32f(guided, guided_32f);

	Mat mat_Ip, mat_I2;   //����I*p��I*I
	multiply(guided_32f, source_32f, mat_Ip);
	multiply(guided_32f, guided_32f, mat_I2);

	Mat mean_p, mean_I, mean_Ip, mean_I2;   //������־�ֵ
	Size win_size(2 * radius + 1, 2 * radius + 1);
	boxFilter(source_32f, mean_p, CV_32F, win_size);
	boxFilter(guided_32f, mean_I, CV_32F, win_size);
	boxFilter(mat_Ip, mean_Ip, CV_32F, win_size);
	boxFilter(mat_I2, mean_I2, CV_32F, win_size);

	Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);//����Ip��Э�����I�ķ���
	Mat var_I = mean_I2 - mean_I.mul(mean_I);
	var_I += epsilon;

	Mat a, b;   //��a��b
	divide(cov_Ip, var_I, a);
	b = mean_p - a.mul(mean_I);

	Mat mean_a, mean_b;  //�԰�������i������a��b��ƽ��
	boxFilter(a, mean_a, CV_32F, win_size);
	boxFilter(b, mean_b, CV_32F, win_size);

	output = mean_a.mul(guided_32f) + mean_b;//������� (depth == CV_32F)
}


void HazeRemoval(Mat& image, Mat& imageRGB)
{
	CV_Assert(!image.empty() && image.channels() == 3);
	Mat fImage;
	image.convertTo(fImage, CV_32FC3, 1.0 / 255, 0);//ͼƬ��һ��

	Mat fImageBorder;
	int hPatch = 15, vPatch = 15;//�趨��С�˲�patch�Ĵ�С,�Ҿ�Ϊ����
	copyMakeBorder(fImage, fImageBorder, vPatch / 2, vPatch / 2, hPatch / 2, hPatch / 2, BORDER_REPLICATE);//����һ����ͼƬ��ӱ߽�
	std::vector<Mat> fImageBorderVector(3);
	split(fImageBorder, fImageBorderVector);//����ͨ��

	Mat darkChannel(image.rows, image.cols, CV_32FC1);//����darkChannel
	double minTemp, minPixel;
	for (unsigned int r = 0; r < darkChannel.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannel.cols; c++)
		{
			minPixel = 1.0;
			for (std::vector<Mat>::iterator it = fImageBorderVector.begin(); it != fImageBorderVector.end(); it++)
			{
				Mat roi(*it, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			darkChannel.at<float>(r, c) = float(minPixel);
		}
	}
	//darkChannel.convertTo(darkChannel8U, CV_8UC1, 255, 0);

	//���A(global atmospheric light),�����darkChannel��ǰtop������ֵ,������ȡֵΪ0.1%
	float top = 0.001;
	float numberTop = top * darkChannel.rows * darkChannel.cols;
	Mat darkChannelVector;
	darkChannelVector = darkChannel.reshape(1, 1);//reshape��һ��������ʾͨ�������ڶ�����ʾ��������
	Mat_<int> darkChannelVectorIndex;
	sortIdx(darkChannelVector, darkChannelVectorIndex, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);//���򣬷�����������

	int count = 0, temp = 0;
	unsigned int x, y; //ӳ��ذ�ͨ��ͼ������λ��
	Mat mask(darkChannel.rows, darkChannel.cols, CV_8UC1);//��������,ע��mask�����ͱ�����CV_8UC1
	for (unsigned int r = 0; r < darkChannelVectorIndex.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannelVectorIndex.cols; c++)
		{
			temp = darkChannelVectorIndex.at<int>(r, c);
			x = temp / darkChannel.cols;
			y = temp % darkChannel.cols;

			if (count < numberTop) {
				mask.at<uchar>(x, y) = 1;
				count++;
			}
			else
				mask.at<uchar>(x, y) = 0;

		}
	}

	std::vector<double> A(3);                //�ֱ��ȡB,G,Rͨ�������Aֵ
	std::vector<Mat> fImageBorderVectorA(3);//�����������t(x)ʱ�����õ����µľ������������ǰ���
	std::vector<double>::iterator itA = A.begin();
	std::vector<Mat>::iterator it = fImageBorderVector.begin();
	std::vector<Mat>::iterator itAA = fImageBorderVectorA.begin();
	for (; it != fImageBorderVector.end() && itA != A.end() && itAA != fImageBorderVectorA.end(); it++, itA++, itAA++)
	{
		Mat roi(*it, Rect(0, 0, darkChannel.cols, darkChannel.rows));
		minMaxLoc(roi, 0, &(*itA), 0, 0, mask);
		(*itAA) = (*it) / (*itA); //ע�⣺����ط��г��ţ�����û���ж��Ƿ����0,*itA������Ŀ����Ժ�С
	}

	//���t(x)
	Mat darkChannelA(darkChannel.rows, darkChannel.cols, CV_32FC1);
	float omega = 0.95;      //������ȡֵΪ0.95
	for (unsigned int r = 0; r < darkChannel.rows; r++)
	{
		for (unsigned int c = 0; c < darkChannel.cols; c++)
		{
			minPixel = 1.0;
			for (itAA = fImageBorderVectorA.begin(); itAA != fImageBorderVectorA.end(); itAA++)
			{
				Mat roi(*itAA, Rect(c, r, hPatch, vPatch));
				minMaxLoc(roi, &minTemp);
				minPixel = min(minPixel, minTemp);
			}
			darkChannelA.at<float>(r, c) = float(minPixel);
		}
	}
	Mat tx1 = 1.0 - omega * darkChannelA;
	Mat tx(darkChannel.rows, darkChannel.cols, CV_32FC1);

	guidedFilter(tx1, tx1, tx, 8, 500);

	namedWindow("tx", CV_WINDOW_AUTOSIZE);
	imshow("tx", tx);


	//���J(x)
	float t0 = 0.1;//������ȡ0.1
	//Mat jx(image.rows, image.cols, CV_32FC3);
	for (size_t r = 0; r < imageRGB.rows; r++)
	{
		for (size_t c = 0; c < imageRGB.cols; c++)
		{
			imageRGB.at<Vec3f>(r, c) = Vec3f((fImage.at<Vec3f>(r, c)[0] - A[0]) / max(tx.at<float>(r, c), t0) + A[0], (fImage.at<Vec3f>(r, c)[1] - A[1]) / max(tx.at<float>(r, c), t0) + A[1], (fImage.at<Vec3f>(r, c)[2] - A[2]) / max(tx.at<float>(r, c), t0) + A[2]);
		}
	}

}


int main()
{
	Mat image = imread("D://360MoveData//Users//MarcYu//Desktop//DIP project//db//IEI2019//H26.jpg");//����0д��Ҷ�ͼ��
	std::cout << 'here' << std::endl;
	if (image.empty())
	{
		std::cout << "��ͼƬʧ��,����" << std::endl;
		system("pause");
		return -1;
	}
	Mat testimage(image.size(), CV_32FC3);
	HazeRemoval(image, testimage);

	namedWindow("ԭͼ", CV_WINDOW_AUTOSIZE);
	namedWindow("�任���ͼ", CV_WINDOW_AUTOSIZE);
	imshow("ԭͼ", image);
	imshow("�任���ͼ", testimage);
	waitKey(0);
	destroyAllWindows();

	system("pause");
	return 0;
}