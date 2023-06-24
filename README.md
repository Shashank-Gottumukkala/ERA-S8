# Normalization Techniques on CIFAR-10 Dataset

This code repository provides an implementation of various normalization techniques, including Batch Normalization (BN), Layer Normalization (LN), and Group Normalization (GN) on the `CIFAR-10` dataset. The `CIFAR-10` dataset consists of `60,000` `32x32` color images in `10` different classes.

## Normalization Techniques
The following normalization techniques are implemented in the code:

### `Batch Normalization (BN):` 
  - Batch normalization is applied to the activations of each layer, normalizing the activations to have zero mean and unit variance.
    
### `Layer Normalization (LN):` 
  - Layer normalization is applied to the activations of each layer, normalizing the activations to have zero mean and unit variance along the channel dimension.
    
### `Group Normalization (GN):` 
  - Group normalization divides the channels into groups and computes mean and variance independently within each group, normalizing the activations.

## File Structure
The code repository has the following file structure:

  - `dataset.py` : This code provides a `DataSet` class that serves as a base class for handling image datasets. It includes functionality for loading and transforming data, as well as displaying examples from the dataset.
      ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/b58723c5-0332-4e9a-98cb-d9c7ec44ae48)

  - `model.py` : This code provides different Convolutional Neural Network (CNN) models implemented using PyTorch, including variations with different normalization techniques.
    
    - The code includes the following CNN models:

      - `Net`: The base CNN model that can be customized with different normalization techniques and options.
      - `GroupNormModel` : A variant of the Net model that uses Group Normalization (GN) for normalization.
      - `LayerNormModel` : A variant of the Net model that uses Layer Normalization (LN) for normalization.
      - `BatchNormModel` : A variant of the Net model that uses Batch Normalization (BN) for normalization.

    - The ConvLayer class serves as a building block for a neural network and follows a specific structure:

      - Input X is passed through a convolutional layer.
      - The output of the convolutional layer is normalized using a normalization technique (optional).
      - The normalized output is added to the original input X using a skip connection (+X).
      - The result is passed through a rectified linear unit (ReLU) activation function.
      - Normalisation, Skip connection and dropout are optional
        
      - ```bazaar
        X -> Convolution -> Normalisation -> +X -> ReLU -> Dropout
        ```
            
      
          ```python
          def forward(self, x):
              x_ = x
              x = self.convlayer(x)
              if self.normlayer is not None:
                  x = self.normlayer(x)
              if self.skip:
                  x += x_
              x = self.activation(x)
              if self.dropout is not None:
                  x = self.dropout(x)
              return x
          ```

      
  - `utis.py` :
     - This code contains classes `Train`, `Test`, `Experiment` that are designed to facilitate training, testing, and conducting experiments with a given model and dataset.
     - The code also includes utility functions such as `get_correct_count` to calculate the number of correct predictions, `get_incorrect_preds` to retrieve incorrect predictions and their corresponding indices.

## Results:
1. Group Normaliztion
   - No of Params: `47,818`
   - Best Training Accuracy : `71.75`
   - Best Test Accuracy : `73.68`
  
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/efb7f613-09b0-48a1-a679-622ad220e7c6)
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/ae8e1ba7-ac6d-4bee-bfc6-4a4265567ab2)

   ### Misclassified Images:
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/9648407e-7efb-475b-ba80-2c8902c70dd1)

2. Batch Normaliztion
   - No of Params: `47,818`
   - Best Training Accuracy : `76.28`
   - Best Test Accuracy : `79.57`
  
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/3a8f3c92-0043-4ef9-9e02-9232cd70a9e3)
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/a44a793f-14c2-4f81-be95-e87b1e4fe8a2)

   ### Misclassified Images:
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/d1f5982c-356f-4f97-8827-8bbc70f23090)

3. Layer Normaliztion
   - No of Params: `47,818`
   - Best Training Accuracy : `66.94`
   - Best Test Accuracy : `68.55`
  
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/4c855354-05f7-4a2f-9270-2f6bda32ce81)
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/298d90ff-2f58-46cd-b63f-7755361f0b1b)


   ### Misclassified Images:
   ![image](https://github.com/Shashank-Gottumukkala/ERA-S8/assets/59787210/68268936-60bd-4c13-bf9e-0063328bc871)






  


