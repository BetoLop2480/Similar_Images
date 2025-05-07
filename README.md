# Similar_Images
This project implements a Pre-trained network for the determination of similar images in the ECSD dataset.


1. We utilize the pre-trained network of RESNET18 for the feature extraction stage. The key aspects to take into account is that the weights of the model must be "freeze", that is, we need to avoid to compute gradient and at the same time we need to elimnate from the model´s architecture the Fully connected layer to avoid the classification stage.
2. Another key point is that we need to set the model to the evaluation mode to deactivate batch_normalization and dropout stages.
3. In the implementation, we select the image to which we search for their most similar randomly or directly.
4. Once the features vectors are calculated, we compute the euclidean distance to determine a similarity index. Any other distance can be used to compute similarity.
5. The results are disposed in a python native strucutre data such as DataFrame, but any other strategy can be used to allocate the similarity index computed.
6. The Top 3 most similar images are displayed on screen.
7. We use the ECSD datset as our data for evaluating this implmentation. 
