import gradio as gr
from skimage.feature import hog
from skimage import exposure, color

def compute_hog_visualisation(image, orientations, pixels_per_cell, cells_per_block):
    """
    Computes the HOG descriptor and returns a visualisation image.
    This function acts as the core of the Gradio interface.
    """
    # Handle empty inputs (e.g., when the app first loads)
    if image is None:
        return None
        
    # Convert to grayscale if the image is in color
    # HOG is primarily calculated on single-channel (grayscale) images
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
        
    # Compute HOG and request the visualisation image
    # visualize=True is important here, since it gives us the actual line segments to draw
    features, hog_image = hog(
        image,
        orientations=orientations,
        pixels_per_cell=(pixels_per_cell, pixels_per_cell),
        cells_per_block=(cells_per_block, cells_per_block),
        visualize=True,
        block_norm='L2-Hys' # Standard block normalisation method (The original paper uses this)
    )
    
    # Rescale the visualisation for better contrast on the screen
    # This makes the gradient directional lines much easier to see
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range='image')
    
    return hog_image_rescaled


# Gradio interface setup
# We use Gradio Blocks for a slightly cleaner layout than the standard Interface
with gr.Blocks(title="HOG Descriptor Simulator") as demo:
    gr.Markdown("# Histogram of Oriented Gradients (HOG) Simulator")
    gr.Markdown("Upload an image and adjust the parameters to see how the HOG descriptor captures shape and gradients.")
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column():
            input_image = gr.Image(label="Original Image", type="numpy")
            
            # Interactive parameters that match HOG theory
            orientations_slider = gr.Slider(minimum=4, maximum=16, step=1, value=9, label="Orientations (Bins)")
            ppc_slider = gr.Slider(minimum=4, maximum=32, step=4, value=8, label="Pixels per Cell")
            cpb_slider = gr.Slider(minimum=1, maximum=4, step=1, value=2, label="Cells per Block")
            
        # Right Column: Output
        with gr.Column():
            output_image = gr.Image(label="HOG Visualisation")
            
    # When any of the inputs change, trigger the function and update the output
    inputs = [input_image, orientations_slider, ppc_slider, cpb_slider]
    
    # Trigger updates on parameter changes
    input_image.change(fn=compute_hog_visualisation, inputs=inputs, outputs=output_image)
    orientations_slider.change(fn=compute_hog_visualisation, inputs=inputs, outputs=output_image)
    ppc_slider.change(fn=compute_hog_visualisation, inputs=inputs, outputs=output_image)
    cpb_slider.change(fn=compute_hog_visualisation, inputs=inputs, outputs=output_image)

    gr.Markdown("### Try these examples:")
    gr.Examples(
        examples=[
            # The example images provided below are sourced from the PnPLO Dataset.
            # Citation: Karthika, N.J. & Saravanan, C. (2020). Addressing False Positives in Pedestrian Detection.
            # Format: [image_path, orientations, pixels_per_cell, cells_per_block]
            ["images/pedestrian_full_body.jpg", 9, 8, 2], # Perfect use case for HOG
            ["images/small_pedestrian_night.jpg", 9, 8, 2],     # Failure mode: HOG struggles with low-light and small objects
            ["images/small_pedestrians.jpg", 9, 8, 2]       # Failure mode: HOG struggles with small objects (e.g. people) overlapping and cluttered backgrounds
        ],
        inputs=[input_image, orientations_slider, ppc_slider, cpb_slider],
        outputs=output_image,
        fn=compute_hog_visualisation,
        cache_examples=True # This pre-computes them so they load instantly!
    )

# Launch the app locally
if __name__ == "__main__":
    demo.launch()