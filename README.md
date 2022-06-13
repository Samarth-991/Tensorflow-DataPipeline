# Tensorflow-DataPipeline

The TFRecord format is a simple format for storing a sequence of binary records.Protocol buffers are a cross-platform, cross-language library for efficient serialization of structured data.Protocol messages are defined by .proto files, these are often the easiest way to understand a message type.

The tf.train.Example message (or protobuf) is a flexible message type that represents a {"string": value} mapping. It is designed for use with TensorFlow and is used throughout the higher-level APIs such as TFX.
The repository have examples on how to create data pipleines using TF-records for Training and Evaluation purposes along with how to train model with TF-Records

## Image Classification
Created a TF-Record file for image classification to classify dog and cat dataset from kaggle. Similar structure can be used for complex operations as well. 

### Train
1. Requirements:
+ Python >= 3.9
+ Tensorflow >= 2.7.0
+ tensorflow-addons >= 0.15.0
2. To train the network on your own dataset, you can put the dataset under the folder **dataset**. Here I have used directory structure:
```
|——dataset
   |——class_name_0.jpeg
   |——class_name_1.jpeg
   |——class_name_2.jpeg
   |——class_name_3.jpeg
```
3. Run the script **python train.py -p <data_path>** to create tf-records for train and validation data and run training.
4. Any of the models available in Tensorflow can be used , check the image size before selecting the Model
## Different input image sizes for different neural networks
<table>
     <tr align="center">
          <th>Type</th>
          <th>Neural Network</th>
          <th>Input Image Size (height * width)</th>
     </tr>
     <tr align="center">
          <td rowspan="3">MobileNet</td>
          <td>MobileNet_V1</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>MobileNet_V2</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>MobileNet_V3</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>EfficientNet</td>
          <td>EfficientNet(B0~B7)</td>
          <td>/</td>
     </tr>
     <tr align="center">
          <td rowspan="2">ResNeXt</td>
          <td>ResNeXt50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNeXt101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="2">SEResNeXt</td>
          <td>SEResNeXt50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SEResNeXt101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="3">Inception</td>
          <td>InceptionV4</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td>Inception_ResNet_V1</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td>Inception_ResNet_V2</td>
          <td>(299 * 299)</td>
     </tr>
     <tr align="center">
          <td rowspan="3">SE_ResNet</td>
          <td>SE_ResNet_50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SE_ResNet_101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>SE_ResNet_152</td>
          <td>(224 * 224)</td>
     </tr>
     </tr align="center">
          <td>SqueezeNet</td>
          <td align="center">SqueezeNet</td>
          <td align="center">(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="4">DenseNet</td>
          <td>DenseNet_121</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_169</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_201</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>DenseNet_269</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ShuffleNetV2</td>
          <td>ShuffleNetV2</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td rowspan="5">ResNet</td>
          <td>ResNet_18</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_34</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_50</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_101</td>
          <td>(224 * 224)</td>
     </tr>
     <tr align="center">
          <td>ResNet_152</td>
          <td>(224 * 224)</td>
     </tr>
</table>
