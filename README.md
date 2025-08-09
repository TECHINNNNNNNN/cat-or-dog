# Cat vs Dog Classifier

A CNN-based image classifier that can tell cats from dogs with 88.7% accuracy. Built this to learn TensorFlow and ended up making something pretty cool!

**Techin Chompooborisuth** | [GitHub](https://github.com/TECHINNNNNNNN) | [LinkedIn](https://www.linkedin.com/in/techin-chompooborisuth-396b19268)

## What it does

Upload a cat or dog image and the model predicts which one it is. I mean, sounds simple enough, but there's actually a lot going on under the hood. The web interface shows confidence scores, feature maps, and lets you batch process multiple images.

## The build

Custom CNN with 4 conv blocks (32→64→128→128 filters). Nothing too fancy, but it gets the job done:
- 150x150 RGB input 
- ~3.4M parameters
- Dropout and data augmentation to prevent overfitting
- Trained on 25k images from Kaggle

I learned that getting good accuracy isn't just about the architecture - data preprocessing and augmentation made a huge difference.

## Quick start

```bash
git clone https://github.com/TECHINNNNNNNN/cat-or-dog.git
cd cat-or-dog
pip install -r requirements.txt
streamlit run app.py
```

Then go to `http://localhost:8501` and upload some images!

## How I built it

Started in Jupyter notebooks experimenting with different architectures. The tricky part was handling the RGBA to RGB conversion - PNG images with transparency were causing issues until I added proper preprocessing.

I also ran into layer naming mismatches when building the visualization features. Keras auto-generates layer names, but I wanted more intuitive names for the UI. Had to map between the actual layer names and display names.

Training took about 30 minutes on Colab's T4 GPU. The model hit 88.7% validation accuracy at epoch 14 - pretty solid for a custom CNN!

## What's in here

```
├── app.py                    # Streamlit web interface  
├── cat_dog_cnn_model.h5      # Trained model weights
├── src/
│   ├── model_utils.py        # Core prediction logic
│   └── visualization.py      # Feature map visualizations
├── notebooks/                # Development notebooks
└── requirements.txt
```

## Features

- **Real-time predictions** with confidence scores
- **Feature map visualization** to see what the CNN learned
- **Batch processing** for multiple images
- **Interactive charts** and confidence gauges

The visualization part was super cool to implement - you can actually see what patterns each convolutional layer picks up on.

## Development notes

Check out `notebooks/cat_dog_classifier_clean.ipynb` for the full training pipeline. The original development notebook shows the messy exploration process - kept it for reference.

## What I learned

- Data augmentation is crucial for preventing overfitting
- Image preprocessing edge cases (like RGBA channels) can break things in subtle ways  
- Feature visualization helps debug what the model actually learned
- Streamlit makes building ML demos really straightforward

## Next improvements

- Transfer learning with ResNet or EfficientNet (should boost accuracy)
- Grad-CAM visualization for better interpretability  
- API endpoints for programmatic access

## License

MIT - see LICENSE file

---

Built by **Techin Chompooborisuth**