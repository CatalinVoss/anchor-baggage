//
//  main.cpp
//  VisualBaggage
//
//  Created by Catalin Voss on 5/11/14.
//  Copyright (c) 2014 Sension, Inc. All rights reserved.
//

// pieces from http://www.codeproject.com/Articles/619039/Bag-of-Features-Descriptor-on-SIFT-Features-with-O
// example of supervised http://aclweb.org/anthology//N/N10/N10-1125.pdf

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;

const string input_path = "/Users/Catalin/Desktop/bag_data/";
const string dict_output = "/Users/Catalin/Desktop/bag_data/dictionary.yml";

void build_dictionary(Mat &out, vector<string> &image_files) {
    // to store the current input image
    Mat input;
    
    //To store the keypoints that will be extracted by SIFT
    vector<KeyPoint> keypoints;
    //To store the SIFT descriptor of current image
    Mat descriptor;
    //To store all the descriptors that are extracted from all the images.
    Mat featuresUnclustered;
    //The SIFT feature extractor and descriptor
    SiftDescriptorExtractor detector;
    
    char filename[255]; // store filename as we roll
    
    // For each image --> build the vocabulary
    for(int f=0; f < image_files.size(); f++){
        //create the file name of an image
        sprintf(filename,"%s%s", input_path.c_str(), image_files[f].c_str());
        //open the file
        input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE); //Load as grayscale
        //detect feature points
        detector.detect(input, keypoints);
        //compute the descriptors for each keypoint
        detector.compute(input, keypoints,descriptor);
        //put the all feature descriptors in a single Mat object
        featuresUnclustered.push_back(descriptor);
        //print the percentage
        if (f%50==0)
            printf("%f percent done\n",100*(float)f/image_files.size());
    }
    
    //Construct BOWKMeansTrainer
    //the number of bags
    int dictionarySize=200;
    //define Term Criteria
    TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
    //retries number
    int retries=1;
    //necessary flags
    int flags=KMEANS_PP_CENTERS;
    //Create the BoW (or BoF) trainer
    BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
    //cluster the feature vectors
    Mat dictionary=bowTrainer.cluster(featuresUnclustered);

    out = dictionary;
}

// Builds bag of words representation of input directory
int main(int argc, const char * argv[]) {
    
    // Load image names
    vector<string>image_files;
    string line; // reusable storage
    ifstream id_file((input_path+"list.txt").c_str());
    while (std::getline(id_file, line))
        image_files.push_back(line);
    
    Mat dictionary;
    
    // To build the dictionary, uncomment:
    //
    build_dictionary(dictionary, image_files);
    // Store the vocabulary
    FileStorage fs(dict_output.c_str(), FileStorage::WRITE);
    fs << "vocabulary" << dictionary;
    fs.release();
    
    // Load dictionary
//    FileStorage fs(dict_output.c_str(), FileStorage::READ);
//    fs["vocabulary"] >> dictionary;
//    fs.release();
    
    //create a nearest neighbor matcher
    Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
    //create Sift feature point extracter
    Ptr<FeatureDetector> detector(new SiftFeatureDetector());
    //create Sift descriptor extractor
    Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
    //create BoF (or BoW) descriptor extractor
    BOWImgDescriptorExtractor bowDE(extractor,matcher);
    //Set the dictionary with the vocabulary we created in the first step
    bowDE.setVocabulary(dictionary);
    
    //To store the image file name
    char filename[100];
    
    
    Mat global_descriptors;
    
    for (int i = 0; i < image_files.size(); i++) {
        sprintf(filename,"%s%s", input_path.c_str(), image_files[i].c_str());
        
        //read the image
        Mat img=imread(filename,CV_LOAD_IMAGE_GRAYSCALE);
        //To store the keypoints that will be extracted by SIFT
        vector<KeyPoint> keypoints;
        //Detect SIFT keypoints (or feature points)
        detector->detect(img,keypoints);
        //To store the BoW (or BoF) representation of the image
        Mat bowDescriptor;
        //extract BoW (or BoF) descriptor from given image
        bowDE.compute(img,keypoints,bowDescriptor);
        
        global_descriptors.push_back(bowDescriptor);
        
        // The bag of words description for a single image.
        //cout << i << "Descriptor: " << bowDescriptor << endl << endl;
        
        if (i % 100 == 0)
            cout << (float)100*i/image_files.size() << " percent done" << endl;
    }
     
    // Write descriptors
    FileStorage fs1("/Users/Catalin/Desktop/descriptor.yml", FileStorage::WRITE);
    fs1 << "descriptors" << global_descriptors;
    fs1.release();
    
    // Read descriptors
    Mat descriptors;
    FileStorage fs2("/Users/Catalin/Desktop/descriptor.yml", FileStorage::READ);
    fs2["descriptors"] >> descriptors;
    fs2.release();
    
    // Write document matrix as CSV
    ofstream outFile;
    outFile.open("/users/Catalin/Desktop/documents.csv");
    
    for (int i = 0; i < descriptors.rows; i++) {
        for (int j = 0; j < descriptors.cols; j++) {
            outFile << descriptors.at<float>(i,j);
            if (j < descriptors.cols-1)
                outFile << ",";
        }
        outFile << endl;
    }
    outFile.close();
    
    return 0;
}

