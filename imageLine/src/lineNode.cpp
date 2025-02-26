#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <nav_msgs/Odometry.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>  // ������������ĺ���
#include <opencv2/imgcodecs.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <algorithm>  // For std::sort
#include <memory>  // For std::unique_ptr



using namespace cv;
using namespace cv::line_descriptor;
using namespace cv::ximgproc;
using namespace std;
using namespace message_filters;

// Function declarations
vector<DMatch> matchLineFeatures(const Mat& descriptors1, const Mat& descriptors2);
void detectAndComputeLineFeatures(const Mat& image, vector<KeyLine>& keylines, Mat& descriptors);
void drawLineFeatures(Mat& image, const vector<KeyLine>& keylines, const Scalar& color);
void drawMatchesOnImage(Mat& combinedImage, const vector<KeyLine>& keylines1, const vector<KeyLine>& keylines2, const vector<DMatch>& matches);

// Global variables
const double PI = 3.14159265358979323846;
vector<KeyLine> prev_keylines;
Mat prev_descriptors;
cv::Mat prev_image;  // ���ڴ洢��һ֡ͼ��
ros::Publisher image_pub, data_pub; // Publisher for output image
bool init = false;
Eigen::Vector3d prev_translation(0.0, 0.0, 0.0);
Eigen::Matrix3d prev_rotation = Eigen::Matrix3d::Identity();
double c_x = 421.99731445, c_y = 397.4869995117, f_x = 286.0103149414, f_y = 285.9841918945, 
       k1 = 0.048112031072378, k2 = -0.04610678181, k3 = -0.01094698999, k4 = 0.00893077440559864 ;  // ����ϵ��

class LineFeatureMatch {
public:
    // ��Ա����
    std::vector<int> prev_indices;  // �洢ǰһ֡������������
    std::vector<int> curr_indices;  // �洢��ǰ֡������������

    std::vector<cv::Point2f> midpoints_prev;  // �洢ǰһ֡�߶��е����������
    std::vector<cv::Point2f> midpoints_curr;  // �洢��ǰ֡�߶��е����������

    // �洢��������
    std::vector<cv::Vec3f> sphere_coords_prev;  // ǰһ֡���е���������
    std::vector<cv::Vec3f> sphere_coords_curr;  // ��ǰ֡���е���������

    // ���캯��
    LineFeatureMatch() {}

    // ��ƥ����������е�洢��ת��Ϊ��λ��������
    void storeMidpointsAndSphereCoords(const std::vector<DMatch>& matches,
                                       const std::vector<KeyLine>& keylines_prev,
                                       const std::vector<KeyLine>& keylines_curr,
                                       const cv::Mat& K) {
        // ��������ƥ��
        for (const auto& match : matches) {
            int prev_idx = match.trainIdx;  // ǰһ֡����������
            int curr_idx = match.queryIdx;  // ��ǰ֡����������

            cv::Point2f start_prev(keylines_prev[prev_idx].startPointX, keylines_prev[prev_idx].startPointY);
            cv::Point2f end_prev(keylines_prev[prev_idx].endPointX, keylines_prev[prev_idx].endPointY);

            cv::Point2f start_curr(keylines_curr[curr_idx].startPointX, keylines_curr[curr_idx].startPointY);
            cv::Point2f end_curr(keylines_curr[curr_idx].endPointX, keylines_curr[curr_idx].endPointY);


            // ��ȡǰһ֡�͵�ǰ֡���߶������յ�
            // cv::Point2f start_prev = keylines_prev[prev_idx].startPoint;
            // cv::Point2f end_prev = keylines_prev[prev_idx].endPoint;

            // cv::Point2f start_curr = keylines_curr[curr_idx].startPoint;
            // cv::Point2f end_curr = keylines_curr[curr_idx].endPoint;

            // �����е�
            cv::Point2f midpoint_prev = computeMidpoint(start_prev, end_prev);
            cv::Point2f midpoint_curr = computeMidpoint(start_curr, end_curr);

            // ���е�ת������λ������
            cv::Vec3f sphere_coord_prev = pixelToSphere(midpoint_prev, K);
            cv::Vec3f sphere_coord_curr = pixelToSphere(midpoint_curr, K);

            // ������洢
            prev_indices.push_back(prev_idx);
            curr_indices.push_back(curr_idx);
            midpoints_prev.push_back(midpoint_prev);
            midpoints_curr.push_back(midpoint_curr);
            sphere_coords_prev.push_back(sphere_coord_prev);
            sphere_coords_curr.push_back(sphere_coord_curr);
        }
    }

    // �����߶��е�
    cv::Point2f computeMidpoint(const cv::Point2f& start, const cv::Point2f& end) {
        return cv::Point2f((start.x + end.x) / 2.0f, (start.y + end.y) / 2.0f);
    }

    // ����������ת��Ϊ��λ�����ϵ� 3D ���꣨�µ�ͶӰ��ʽ��
    cv::Vec3f pixelToSphere(const cv::Point2f& pixel, const cv::Mat& K) {
        // ����ڲξ��� K �ṩ�˽��� fx, fy �Լ����� cx, cy
        double fx = K.at<double>(0, 0);
        double fy = K.at<double>(1, 1);
        double cx = K.at<double>(0, 2);
        double cy = K.at<double>(1, 2);

        // ����������ת��Ϊ��һ��ͼ������
        double x_prime = (pixel.x - cx) / fx;
        double y_prime = (pixel.y - cy) / fy;

        // ������������
        cv::Vec3d direction(x_prime, y_prime, 1.0);

        // ��һ����������
        cv::Vec3d normalized_direction = direction / cv::norm(direction);

        // ���ع�һ���ķ�������
        return normalized_direction;
    }

    // ��������ֱ��֮�����̾��룬���ж�������Ƿ����������췽����
    double computeShortestDistance(
        const Eigen::Vector3d& P1, const Eigen::Vector3d& P2, // ��һ��ֱ�ߵ�������
        const Eigen::Vector3d& Q1, const Eigen::Vector3d& Q2)  // �ڶ���ֱ�ߵ�������
    {
        // ��������
        Eigen::Vector3d u = P2 - P1; // ��һ��ֱ�ߵķ�������
        Eigen::Vector3d v = Q2 - Q1; // �ڶ���ֱ�ߵķ�������
        Eigen::Vector3d w = P1 - Q1;

        double a = u.dot(u); // u ? u
        double b = u.dot(v); // u ? v
        double c = v.dot(v); // v ? v
        double d = u.dot(w); // u ? w
        double e = v.dot(w); // v ? w
        double D = a * c - b * b; // �����ĸ

        // ��ʼ������
        double s, t;

        // �ж��Ƿ�ƽ��
        if (std::abs(D) < 1e-6) { // D �ӽ��� 0����Ϊֱ��ƽ��
            // ������������������ȡ s = 0�������Ӧ�� t
            s = 0.0;
            t = (b > c ? d / b : e / c);
        } else {
            s = (b * e - c * d) / D;
            t = (a * e - b * d) / D;
        }

        // �жϲ����Ƿ��� [0, +��) ��Χ�ڣ����ж�������Ƿ����������췽����
        bool sInDirection = (s >= 0.0);
        bool tInDirection = (t >= 0.0);

        // �������㲻���������췽���ϣ����� -1
        // if (!sInDirection || !tInDirection) {
        //     return -1.0;
        // }

        //���������꣨��ѡ�������Ҫ�������������꣬���Լ��㣩
        Eigen::Vector3d closestPointOnLine1 = P1 + s * u;
        Eigen::Vector3d closestPointOnLine2 = Q1 + t * v;

        // ������̾���
        Eigen::Vector3d diff = (P1 + s * u) - (Q1 + t * v);
        double distance = diff.norm();

        return distance;
    }

    // ���ֺ���
    double computeScore(std::vector<DMatch>& matches,  // Note: Take matches by non-const reference
                        const Eigen::Matrix3d& delta_rotation,
                        const Eigen::Vector3d& delta_translation,
                        double dist_th) 
    {
        if (sphere_coords_prev.size() != sphere_coords_curr.size()) {
            std::cerr << "Error: Size mismatch between previous and current sphere coordinates." << std::endl;
            return -1.0;
        }

        // �� matches ����ƥ������������ȷ��ǰ 80% ��ƥ����õ�
        std::sort(matches.begin(), matches.end(), [](const DMatch& a, const DMatch& b) {
            return a.distance < b.distance;
        });

        // �ۼӷ���
        double distance_sum = 0.0;
        int count = 0;

        // ����ǰ80%ƥ����Ŀ
        size_t n_matches = matches.size();
        size_t n_top_matches = static_cast<size_t>(n_matches * 1.0);

        // ����ǰ80%��ƥ�䣬���б任���ۼ�
        for (size_t i = 0; i < n_top_matches; ++i) {
            // ��ǰ֡����������
            Eigen::Vector3d curr_coord(sphere_coords_curr[i][0], sphere_coords_curr[i][1], sphere_coords_curr[i][2]);

            // �Ե�ǰ֡������������б任
    // ���Ƕ�ת��Ϊ����
    double roll = 1 * M_PI / 180.0;  // Χ�� X �����ת�Ƕ�
    double pitch = 1 * M_PI / 180.0; // Χ�� Y �����ת�Ƕ�
    double yaw = 1 * M_PI / 180.0;   // Χ�� Z �����ת�Ƕ�

    // ������ X �����ת����
    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0,
          0, cos(roll), -sin(roll),
          0, sin(roll), cos(roll);

    // ������ Y �����ת����
    Eigen::Matrix3d Ry;
    Ry << cos(pitch), 0, sin(pitch),
          0, 1, 0,
          -sin(pitch), 0, cos(pitch);

    // ������ Z �����ת����
    Eigen::Matrix3d Rz;
    Rz << cos(yaw), -sin(yaw), 0,
          sin(yaw), cos(yaw), 0,
          0, 0, 1;

    // ������ת������������ Z��Y��X ����ת�Ľ��
    Eigen::Matrix3d delta_rotation_test = Rz * Ry * Rx;
             
    Eigen::Vector3d delta_translation_test(0.03, 0.02, 0.012);

            Eigen::Vector3d curr_coord_transformed_P2 = delta_rotation * curr_coord + delta_translation;
            Eigen::Vector3d curr_coord_transformed_P1 = delta_translation;

            // ǰһ֡����������
            Eigen::Vector3d prev_coord_P2(sphere_coords_prev[i][0], sphere_coords_prev[i][1], sphere_coords_prev[i][2]);
            Eigen::Vector3d prev_coord_P1(0.0 , 0.0, 0.0);

            double distance = computeShortestDistance(curr_coord_transformed_P1, curr_coord_transformed_P2,prev_coord_P1, prev_coord_P2);

            if (distance >= 0.0) 
            {
                // �����̾���
                std::cout << "The shortest dist: " << distance << std::endl;
                // �ۼӾ���
                distance_sum += std::abs(distance);
                count++;
            } 
            else {
                // ��� -1����ʾ����㲻���������췽����
                std::cout << "Wrong match or wrong odom." << distance << std::endl;
            }


            // // ��һ������
            // Eigen::Vector3d curr_normalized = curr_coord_transformed.normalized();
            // Eigen::Vector3d prev_normalized = prev_coord.normalized();
            // // ������
            // double dot_product = curr_normalized.dot(prev_normalized);

            // // Ϊ��ֹ��ֵ���³��� [-1, 1] ��Χ�����мн�
            // double dot_product_clamped = std::max(-1.0, std::min(1.0, dot_product));

            // // ����нǣ���λ�����ȣ�
            // double angle = std::acos(dot_product_clamped);
            
        }

        // �����ĸ
        double denominator = count * dist_th;

        // �������յ�����
        double score = (denominator > 0) ? (distance_sum / denominator) : -1.0;

        return score;
    }

    // ��ӡƥ����Ϣ
    void printMatchesInfo() const {
        for (size_t i = 0; i < prev_indices.size(); ++i) {
            std::cout << "Match " << i + 1 << ":\n";
            std::cout << "  Previous frame index: " << prev_indices[i] << "\n";
            std::cout << "  Current frame index: " << curr_indices[i] << "\n";
            std::cout << "  Midpoint in previous frame: (" << midpoints_prev[i].x << ", " << midpoints_prev[i].y << ")\n";
            std::cout << "  Midpoint in current frame: (" << midpoints_curr[i].x << ", " << midpoints_curr[i].y << ")\n";
            std::cout << "  Sphere coordinate in previous frame: [" << sphere_coords_prev[i][0] << ", " << sphere_coords_prev[i][1] << ", " << sphere_coords_prev[i][2] << "]\n";
            std::cout << "  Sphere coordinate in current frame: [" << sphere_coords_curr[i][0] << ", " << sphere_coords_curr[i][1] << ", " << sphere_coords_curr[i][2] << "]\n";
        }
    }
};

vector<DMatch> matchLineFeatures(const Mat& descriptors1, const Mat& descriptors2) {
    // ����һ��ƥ����������ʹ�� BruteForce-Hamming ������ƥ��
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // ��������������Ƿ�Ϊ��
    if (descriptors1.empty() || descriptors2.empty()) {
        ROS_WARN("One of the descriptor matrices is empty, skipping matching.");
        return {};  // ���Ϊ�գ�ֱ�ӷ��ؿ�ƥ����
    }

    // ���� KNN ƥ�� (k=2)
    vector<vector<DMatch>> knn_matches;
    matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    vector<DMatch> good_matches;
    const float ratio_thresh = 0.75f;  // Ratio test ��ֵ

    // �������е� KNN ƥ����
    for (size_t i = 0; i < knn_matches.size(); i++) {
        // ȷ��ÿ��ƥ������������ƥ���
        if (knn_matches[i].size() < 2) continue;

        const DMatch& bestMatch = knn_matches[i][0];  // ��õ�ƥ��
        const DMatch& secondBestMatch = knn_matches[i][1];  // �κõ�ƥ��

        // Ӧ�� ratio test��ȷ����õ�ƥ���������ڴκõ�ƥ��
        if (bestMatch.distance < ratio_thresh * secondBestMatch.distance) {
            good_matches.push_back(bestMatch);  // ͨ�� ratio test ��ƥ�䱣��
        }
    }

    return good_matches;  // ����ͨ�� ratio test ����Чƥ��
}


// Function to detect line features using FastLineDetector (remains the same)
void detectAndComputeLineFeatures(const Mat& image, vector<KeyLine>& keylines, Mat& descriptors) {
    // Create an LSDDetector
    Ptr<line_descriptor::LSDDetector> lsd = line_descriptor::LSDDetector::createLSDDetector();
    if (lsd.empty()) {
        ROS_ERROR("Failed to create LSDDetector!");
        return;
    }

    // Detect lines using LSDDetector
    lsd->detect(image, keylines, 1.2, 1);  // Detect lines and store them directly in 'keylines'

    ROS_INFO("Number of line features detected: %lu", keylines.size());

    // Filter lines with a length less than 50 pixels
    vector<KeyLine> filtered_keylines;
    for (const auto& keyline : keylines) {
        if (keyline.lineLength >= 30) {
            filtered_keylines.push_back(keyline);
        }
    }

    ROS_INFO("Number of line features after filtering: %lu", filtered_keylines.size());

    // If not enough keylines, skip descriptor computation
    if (filtered_keylines.size() < 5) {
        ROS_WARN("Not enough keylines for descriptor computation, skipping...");
        return;
    }

    Ptr<BinaryDescriptor> bd = BinaryDescriptor::createBinaryDescriptor();
    if (bd.empty()) {
        ROS_ERROR("Failed to create BinaryDescriptor!");
        return;
    }

    try {
        ROS_INFO("Computing descriptors...");
        bd->compute(image, filtered_keylines, descriptors);
        ROS_INFO("Descriptor computation completed.");
    } catch (const cv::Exception& e) {
        ROS_ERROR("Exception during descriptor computation: %s", e.what());
    }

    // Update keylines with filtered keylines
    keylines = filtered_keylines;
}

// Function to draw line features on an image (remains the same)
void drawLineFeatures(Mat& image, const vector<KeyLine>& keylines, const Scalar& color) {
    for (const auto& keyline : keylines) 
    {
        line(image, Point2f(keyline.startPointX, keyline.startPointY), 
             Point2f(keyline.endPointX, keyline.endPointY), color, 2);
    }
}

// Function to draw matches on a combined image (remains the same)
void drawMatchesOnImage(Mat& combinedImage, const vector<KeyLine>& keylines1, const vector<KeyLine>& keylines2, const vector<DMatch>& matches) {
    for (const auto& match : matches) {
        int queryIdx = match.queryIdx;
        int trainIdx = match.trainIdx;

        Point2f pt1 = Point2f(keylines1[queryIdx].startPointX, keylines1[queryIdx].startPointY);
        Point2f pt2 = Point2f(keylines2[trainIdx].startPointX + combinedImage.cols / 2, keylines2[trainIdx].startPointY); // Offset for second image

        line(combinedImage, pt1, pt2, Scalar(0, 255, 0), 1);
    }
}

// ʹ�� Kannala-Brandt ģ�ͽ���������ת��Ϊ��������ĺ���
// Eigen::Vector3d PixelToSphere_KannalaBrandt(double u, double v)
// {
//     // ��һ������һ����������
//     double x = (u - c_x) / f_x;
//     double y = (v - c_y) / f_y;

//     // ���㾶����� rd
//     double rd = std::sqrt(x * x + y * y);

//     // ��� rd �ǳ�С��������Ϊ theta Ҳ�ǳ�С��ֱ����Ϊ rd
//     double theta = rd;

//     // �ڶ�����ͨ������������������� theta
//     // ������������������������
//     const int max_iterations = 10;
//     const double tolerance = 1e-10;

//     for (int i = 0; i < max_iterations; ++i)
//     {
//         // ���� theta �ĸ�����
//         double theta2 = theta * theta;
//         double theta3 = theta2 * theta;
//         double theta4 = theta2 * theta2;
//         double theta5 = theta4 * theta;
//         double theta6 = theta4 * theta2;
//         double theta7 = theta6 * theta;
//         double theta8 = theta4 * theta4;
//         double theta9 = theta8 * theta;

//         // ��������ľ������
//         double r_distorted = theta + k1 * theta3 + k2 * theta5 + k3 * theta7 + k4 * theta9;

//         // �����ֵ
//         double delta_r = r_distorted - rd;
//         if (std::abs(delta_r) < tolerance)
//             break;  // ���㾫��Ҫ���˳�����

//         // ���㵼��
//         double derivative = 1 + 3 * k1 * theta2 + 5 * k2 * theta4 + 7 * k3 * theta6 + 9 * k4 * theta8;

//         // ���� theta ֵ
//         theta -= delta_r / derivative;
//     }

//     // �����������㷽λ�� phi
//     double phi = std::atan2(y, x);

//     // ���Ĳ��������������꣨��λ����������
//     double sin_theta = std::sin(theta);
//     double v_x = sin_theta * std::cos(phi);
//     double v_y = sin_theta * std::sin(phi);
//     double v_z = std::cos(theta);

//     // ���� Eigen::Vector3d
//     return Eigen::Vector3d(v_x, v_y, v_z);
// }

// Synchronized callback function for both image and odometry
void callback(const sensor_msgs::ImageConstPtr& image_msg, const nav_msgs::OdometryConstPtr& odom_msg) {

    //motion
    // Extract translation from odom_msg
    Eigen::Vector3d curr_translation(odom_msg->pose.pose.position.x,
                                odom_msg->pose.pose.position.y,
                                odom_msg->pose.pose.position.z);
    
    // Extract rotation from odom_msg (Quaternion to Rotation Matrix)
    Eigen::Quaterniond quaternion(odom_msg->pose.pose.orientation.w,
                                  odom_msg->pose.pose.orientation.x,
                                  odom_msg->pose.pose.orientation.y,
                                  odom_msg->pose.pose.orientation.z);
    Eigen::Matrix3d curr_rotation = quaternion.toRotationMatrix();

    // Get the frame transformation
    Eigen::Matrix3d delta_rotation = prev_rotation.inverse() * curr_rotation;
    Eigen::Vector3d delta_translation = prev_rotation.inverse() * (curr_translation - prev_translation);

    // ���� delta_rotation ����ת�Ƕ�
    double angle = std::acos((delta_rotation.trace() - 1) / 2); // ������ת�����Ӧ����ת�Ƕ�

    // ���� prev_translation ��ģֵ
    double translation_norm = delta_translation.norm();

    // ����Ƕ�С��30�� ���� prev_translation ��ģֵС��0.5m������
    if (angle < M_PI / 18 && translation_norm < 0.3) {
        return;
    }

    ros::Time start = ros::Time::now();
    ROS_INFO("Received a synchronized image and odometry at time: %f", start.toSec());

    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    // Fish-eye camera undistortion
    Mat undistorted_image;

    Mat K = (Mat_<double>(3,3) << f_x, 0, c_x, 0, f_y, c_y, 0, 0, 1);  // Camera intrinsic matrix
    Mat D = (Mat_<double>(4,1) << k1, k2, k3, k4);  // Distortion coefficients

    // ���ͼ��ߴ�
    cv::Size image_size = cv_ptr->image.size();

    // ��ת���� R��ʹ�õ�λ�����ʾ����ת
    cv::Mat R = cv::Mat::eye(3, 3, CV_64F);

    // �µ�����ڲξ��� P�����Ը�����Ҫ�����ӳ�
    cv::Mat P = K.clone();

    // ��ʼ��ȥ�����У��ӳ��
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(K, D, R, P, image_size, CV_16SC2, map1, map2);

    // Ӧ��ӳ�䣬�õ�У�����ͼ��
    cv::remap(cv_ptr->image, undistorted_image, map1, map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    // Convert to grayscale for feature detection
    Mat image;
    cvtColor(undistorted_image, image, COLOR_BGR2GRAY);

    if (image.empty()) {
        ROS_ERROR("Received an empty image!");
        return;
    }

    vector<KeyLine> curr_keylines;
    Mat curr_descriptors;
    detectAndComputeLineFeatures(image, curr_keylines, curr_descriptors);

    int n1 = curr_keylines.size();
    int n2 = prev_keylines.size();

    // Create a combined image (left: previous frame, right: current frame)
    Mat combinedImage(undistorted_image.rows, undistorted_image.cols * 2, CV_8UC3);

    // Convert grayscale to BGR for color drawing of current frame features
    cvtColor(image, image, COLOR_GRAY2BGR);

    Mat left(combinedImage, Rect(0, 0, prev_image.cols, prev_image.rows));

    // Copy previous image to the left, current image to the right
    if (!prev_image.empty()) {
        prev_image.copyTo(left);  // Left: previous frame
    }

    Mat right(combinedImage, Rect(image.cols, 0, image.cols, image.rows));
    undistorted_image.copyTo(right);  // Right: current frame

    // Draw features
    drawLineFeatures(left, prev_keylines, Scalar(0, 0, 255));  // Draw previous frame lines in red
    drawLineFeatures(right, curr_keylines, Scalar(255, 0, 0)); // Draw current frame lines in blue

    // Perform matching if descriptors are available
    if (!prev_keylines.empty() && !prev_descriptors.empty() && !curr_descriptors.empty()) {
        ROS_INFO("Matching line features...");
        vector<DMatch> matches = matchLineFeatures(prev_descriptors, curr_descriptors);
        ROS_INFO("Number of matches: %lu", matches.size());

        // Draw matches on the combined image if any exist
        if (!matches.empty()) {
            drawMatchesOnImage(combinedImage, prev_keylines, curr_keylines, matches);
            // ����ڲξ��� (fx, fy, cx, cy)
            double fx = P.at<double>(0, 0);
            double fy = P.at<double>(1, 1);
            double cx = P.at<double>(0, 2);
            double cy = P.at<double>(1, 2);
            cv::Mat K_2 = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
            // ���������
            LineFeatureMatch lineFeatureMatch;
            // ��ƥ����������е�洢��ת��Ϊ��������
            lineFeatureMatch.storeMidpointsAndSphereCoords(matches, prev_keylines, curr_keylines, K_2);
            // ��ӡƥ����Ϣ
            lineFeatureMatch.printMatchesInfo();
            // �������ֵ���ֵ
            double dist_th = 0.0075;
            // ����ƥ������
            double score = lineFeatureMatch.computeScore(matches, delta_rotation, delta_translation, dist_th);

            // ��������ʼ��һ�� nav_msgs::Odometry ��Ϣ
            nav_msgs::Odometry odom_msg;
            odom_msg.header.stamp = odom_msg.header.stamp;
            odom_msg.pose.pose.position.x = n1;
            odom_msg.pose.pose.position.y = n2;
            odom_msg.pose.pose.position.z = score;

            data_pub.publish(odom_msg);

            // ������ֽ��
            std::cout << "Matching Score: " << score << std::endl;
        }
        } else {
            ROS_INFO("Skipping matching: Not enough data.");
        }

    // Publish the combined image
    ROS_INFO("Publishing combined image...");
    sensor_msgs::ImagePtr output_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", combinedImage).toImageMsg();
    image_pub.publish(output_msg);

    ros::Time end = ros::Time::now();
    ROS_INFO("Image processing took: %f seconds", (end - start).toSec());

    // Update
    prev_keylines = curr_keylines;
    prev_descriptors = curr_descriptors.clone();
    prev_image = undistorted_image.clone();  // Store current undistorted image for the next iteration

    prev_translation = curr_translation;
    prev_rotation = curr_rotation;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "line_feature_matching");
    ros::NodeHandle nh;

    // Initialize the publisher for visualization markers
    image_pub = nh.advertise<sensor_msgs::Image>("line_feature_image", 1);
    data_pub = nh.advertise<nav_msgs::Odometry>("test_data", 5);

    // Subscribers for image and odometry topics
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/fisheye1/image_raw", 10);
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub(nh, "/camera/odom/sample", 10);

    // Set your custom time_threshold (e.g., 10)
    int time_threshold = 10; // You can modify this value or use a ROS parameter to set it dynamically

    // ApproximateTime policy for synchronization
    typedef sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry> SyncPolicy;
    std::unique_ptr<Synchronizer<SyncPolicy>> sync = std::make_unique<Synchronizer<SyncPolicy>>(SyncPolicy(time_threshold), image_sub, odom_sub);
    sync->registerCallback(boost::bind(&callback, _1, _2));

    ros::spin();
    return 0;
}
