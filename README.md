# Image-Restoration
# Image-Restoration

## Project Overview
This project focuses on image restoration using advanced deep learning techniques. The main goal is to enhance and restore the quality of degraded images through the implementation of state-of-the-art deep learning models. The project was developed as part of an academic requirement at the University Euromed of Fez.

## Project Contributors
- Kaoutar Lakdim
- Chaimae Biyaye
- Anas Belmessid
- Monsef Berjaoui
- Manal El Kassim
- Olaya Latoubi

Supervised by:
- Dr. Alae Ammour

## Abstract
The project introduces robust image denoising models such as DnCNN (Deep Convolutional Neural Network). These models leverage deep network architectures comprising multiple convolutional layers, batch normalization, and ReLU activation to effectively learn and remove intricate noise patterns from input images.

## Acknowledgments
We express our sincere gratitude to our supervisor, Dr. Alae Ammour, for his invaluable support throughout the project. His expertise, patience, and guidance have been instrumental in our success.

## Project Structure
### Introduction
- Definition and importance of image restoration
- Overview of degradation sources and types of image noise

### Literature Review
- Classical image restoration techniques
- Advances in deep learning for image restoration

### Theoretical Basis
- Multilayer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Denoising Convolutional Neural Network (DnCNN)
- Convolutional Vision Transformer (CVT)
- PixelCNN

### Methodology
- Data description (using the DIV2K dataset)
- Model architecture and training

### Results and Discussion
- Comparison with previous models
- Performance metrics

### Conclusion
- Summary of findings
- Future work and potential improvements

## Detailed Methodology
### Data Description
The DIV2K dataset, which includes 1,000 high-quality images, was used for training and evaluation. The dataset is divided into training, validation, and testing subsets. The images encompass various scenes and types of degradations to simulate real-world conditions.

### Model Architecture
- **DnCNN**: A specialized CNN designed for denoising tasks. It employs deep convolutional layers and residual learning to effectively suppress noise while preserving image details.
- **CVT**: Combines CNNs and transformers to capture both local and global information for image restoration.
- **PixCNN**: An autoregressive model that generates high-quality restored images by capturing dependencies between pixel values.

### Training
The models were trained using pairs of noisy and clean images. Optimization techniques such as stochastic gradient descent (SGD) and Adam optimizer were used to minimize loss functions like mean squared error (MSE) and improve the models' denoising performance.

## Results
The DnCNN model demonstrated superior performance in reducing image noise, preserving essential details, and achieving high PSNR and SSIM values compared to other techniques.

## Conclusion
The project successfully developed a robust image restoration model using deep learning techniques. The results indicate significant improvements in image quality, with potential applications in various fields such as medical imaging, surveillance, and digital photography.

## Future Work
Future research could focus on further optimizing the models, exploring additional datasets, and integrating more advanced architectures to enhance the restoration capabilities.

## Repository Contents
- `data/`: Contains the DIV2K dataset used for training and evaluation.
- `models/`: Includes the trained models and their configurations.
- `scripts/`: Contains the training and evaluation scripts.
- `results/`: Stores the output images and performance metrics.

## How to Use
1. Clone the repository.
2. Download and extract the DIV2K dataset into the `data/` directory.
3. Run the training script: `python scripts/train.py`
4. Evaluate the models: `python scripts/evaluate.py`
5. View the results in the `results/` directory.

## References
1. [Deep Residual Network with Sparse Feedback for Image Restoration](https://www.mdpi.com/2076-3417/8/12/2417)
2. [Low-dose CT Image Restoration using generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S2352914820306183)
3. [Denoising Prior Driven Deep Neural Network for Image Restoration](https://arxiv.org/pdf/1801.06756.pdf)
4. [Residual Non-Local Attention Networks for Image Restoration](https://arxiv.org/pdf/1903.10082.pdf)

For more detailed information, please refer to the full project report.

You can access the repository and view the complete project [here](https://github.com/anasbelmessid1/AutoML--Ecommerce/new/main?filename=README.md).
