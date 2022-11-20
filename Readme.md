# Age-and-Gender-Detection

<h2>Objective :</h2>
<p>To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture or through webcam.</p>


<h2>About the Project :</h2>
<p>In this Python Project, I had used Deep Learning to accurately identify the gender and age of a person using webcam. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 3), (3 – 7), (7 – 14), (14 – 22), (22 – 35), (35 – 44), (44 – 57), (57 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age using the webcam because of factors like makeup, lighting, obstructions, and facial expressions. And so, I made this a classification problem instead of making it one of regression.</p>

<h2>Additional Python Libraries Required :</h2>
<ul>
  <li>OpenCV</li>
  
       pip install opencv-python
</ul>
<ul>
 <li>argparse</li>
  
       pip install argparse
</ul>
<ul>
  <li>Numpy</li>
  
       pip install numpy
</ul>
<ul>
 <li>Matplotlib</li>
  
       pip install matplotlib
</ul>

<h2>The contents of this Project :</h2>
<ul>
  <li>age.caffemodel</li>
  <li>age.prototxt</li>
  <li>face_detector.pb</li>
  <li>face_detector.pbtxt</li>
  <li>gender.caffemodel</li>
  <li>gender.prototxt</li>
  <li>Age_detection.py</li>
  <li>Wait.jpeg</li>
 </ul>
 
 
  <p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>


 <h2>Waiting Window: </h2>
  <p>This windows is just for visual presentation and it buys you time to set yourself</p>
  <img src= "Wait.jpeg">
 <h2>Usage :</h2>
 <ul>
  <li>Download my Repository</li>
  <li>Search for Age_detection.py and open it using your preferred code editor.</li>
  <li>Set the path of age.caffemodel, age.prototxt, face_detector.pb, face_detector.pbtxt, gender.caffemodel & gender.prototxt </li>
  <li>Open your Command Prompt or Terminal and change directory to the folder where all the files are present.</li>
  
  <li><b> For Detecting Gender and Age using a Webcam </b> Use Command :</li>
  
      python3 Age_detection.py
</ul>
<ul>
  <li>Press <b>Ctrl + C</b> to stop the program execution.</li>
</ul>

<h2>Example :</h2>

    >python3  Age_detection.py
    Gender: Male
    Age: 17-22 years
    
<img src= "Output/Screenshot from 2022-10-08 23-14-33.png">
