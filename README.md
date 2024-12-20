# Music Leitmotif Detection Using Autoencoders

This notebook demonstrates an approach for detecting leitmotifs (recurring musical themes) in audio files using an autoencoder neural network. The audio is processed in segments, and the model learns compact representations of these segments. By analyzing the similarity between these representations, the notebook detects and visualizes the presence of leitmotifs.

## Requirements

The following Python libraries are required:

- `tensorflow` — for building and training the autoencoder model.
- `scikit-learn` — for clustering, similarity measures, and data preprocessing.
- `librosa` — for loading and processing audio files.
- `numpy` — for numerical computations.
- `matplotlib` — for visualizing the results.
- `tsne` — for reducing the dimensionality of embeddings for visualization.

You can install the required dependencies using:

```bash
pip install tensorflow scikit-learn librosa matplotlib numpy
```

## Overview

### 1. **Audio Processing**

The audio files are loaded using `librosa`, and each file is split into segments. Segment duration is defined for each audio file in the `SEGMENT_DURATIONS` dictionary. Each segment is then standardized using `StandardScaler` for better training performance.

### 2. **Autoencoder Architecture**

The notebook constructs and trains an autoencoder model with the following layers:
- **Encoder**: A series of dense layers that compress the input data into a smaller latent space.
- **Decoder**: A series of layers that attempt to reconstruct the original data from the latent space.

The autoencoder is trained using the Mean Squared Error (MSE) loss function and uses early stopping to prevent overfitting.

### 3. **Leitmotif Detection**

Once the autoencoder is trained, the encoder is used to extract embeddings for each segment. Cosine similarity between embeddings is calculated to assess the similarity between segments. A leitmotif is identified when a segment's embedding is similar to a reference (e.g., the first segment).

### 4. **Visualization**

- **Loss Curve**: Visualizes the training and validation loss during the training process.
- **Embedding Visualization**: Uses t-SNE to reduce the dimensionality of the learned embeddings and visualizes them in 2D.
- **Cosine Similarity Matrix**: A heatmap showing the cosine similarity between the segment embeddings.
- **Leitmotif Detection Plot**: A graph showing the similarity over time with detected leitmotif regions highlighted.

### 5. **Song Similarity**

After detecting the leitmotif in each song, the notebook computes the cosine similarity between the leitmotif embeddings from different songs to show how similar their motifs are.

## How to Use

1. **Upload Your Audio Files**  
   Place your audio files (in `.mp3` or `.wav` format) into the working directory or specify the path to the directory containing your audio files.

2. **Run the Notebook**  
   - Open the notebook in Jupyter Notebook or Jupyter Lab.
   - Run the notebook cells sequentially to process your audio files and detect leitmotifs.
   
3. **View the Results**  
   The notebook will output:
   - A training loss curve showing the loss over epochs.
   - A 2D visualization of the embeddings using t-SNE.
   - A heatmap of the cosine similarity matrix between segments.
   - A plot showing the similarity over time and the regions where the leitmotif is detected.
   - The start and end times of the detected leitmotif.
   - The cosine similarity between leitmotifs of different songs.

## Configuration

- **Segment Duration**  
  The segment durations are defined in the `SEGMENT_DURATIONS` dictionary. You can adjust the duration for each audio file. If not specified, the default segment duration is 2 seconds.

- **Autoencoder Architecture**  
  You can modify the architecture of the autoencoder by adjusting the number of layers, neurons in each layer, or the regularization parameters in the `build_autoencoder` function.

- **Leitmotif Detection Threshold**  
  The threshold for detecting leitmotifs based on similarity can be adjusted by modifying the `threshold` argument in the `detect_leitmotif` function.

## Example Output

The notebook will produce the following:

1. **Training Loss Curve**: A plot showing the training and validation loss during autoencoder training.
2. **t-SNE Plot**: A 2D scatter plot representing the embeddings learned by the encoder.
3. **Cosine Similarity Matrix**: A heatmap showing the pairwise similarity between segment embeddings.
4. **Leitmotif Similarity Plot**: A plot displaying how the similarity of segments with the reference leitmotif changes over time, with regions of detected leitmotifs shaded.
5. **Leitmotif Times**: The start and end times (in seconds) of the detected leitmotif regions.
6. **Song-to-Song Similarity**: The cosine similarity between the leitmotif embeddings of different songs is printed.

## Customization

- **Segment Duration**: Modify the `SEGMENT_DURATIONS` dictionary to specify the duration of each audio segment in seconds for different audio files.
  
- **Model Architecture**: You can change the number of layers, neurons per layer, and other hyperparameters of the autoencoder by modifying the `build_autoencoder` function.

- **Leitmotif Threshold**: Adjust the threshold used for detecting leitmotifs by changing the `threshold` parameter in the `detect_leitmotif` function.

- **Audio Files**: Add or remove audio files from the directory. The notebook will automatically process any `.mp3` or `.wav` files in the specified directory.

## Example Files

- **audio1.mp3, audio2.mp3, audio3.mp3**: Example audio files (in MP3 format) to test the process. You can replace these files with your own audio files.

## Notes

- This notebook assumes the audio files are available in the working directory or specified directory. If you want, you can search on youtube for any tracks in mind and use an mp3 converter to make your audio files. make sure you rename them in the dictionary of segments accordingly!
